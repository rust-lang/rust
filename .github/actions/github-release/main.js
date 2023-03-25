// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const core = require('@actions/core');
const path = require("path");
const fs = require("fs");
const github = require('@actions/github');
const glob = require('glob');

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

async function runOnce() {
  // Load all our inputs and env vars. Note that `getInput` reads from `INPUT_*`
  const files = core.getInput('files');
  const token = core.getInput('token');
  const slug = process.env.GITHUB_REPOSITORY;
  const owner = slug.split('/')[0];
  const repo = slug.split('/')[1];
  const sha = process.env.GITHUB_SHA;
  let name = 'dev';
  if (process.env.GITHUB_REF.startsWith('refs/tags/v')) {
    name = process.env.GITHUB_REF.substring(10);
  }

  core.info(`files: ${files}`);
  core.info(`name: ${name}`);
  core.info(`token: ${token}`);

  const octokit = github.getOctokit(token);

  // For the `dev` release we may need to update the tag to point to the new
  // commit on this branch. All other names should already have tags associated
  // with them.
  if (name == 'dev') {
    let tag = null;
    try {
      tag = await octokit.request("GET /repos/:owner/:repo/git/refs/tags/:name", { owner, repo, name });
      core.info(`found existing tag`);
      console.log("tag: ", JSON.stringify(tag.data, null, 2));
    } catch (e) {
      // ignore if this tag doesn't exist
      core.info(`no existing tag found`);
    }

    if (tag === null || tag.data.object.sha !== sha) {
      core.info(`updating existing tag or creating new one`);

      try {
        core.info(`updating dev tag`);
        await octokit.rest.git.updateRef({
          owner,
          repo,
          ref: 'tags/dev',
          sha,
          force: true,
        });
      } catch (e) {
        console.log("ERROR: ", JSON.stringify(e.data, null, 2));
        core.info(`creating dev tag`);
        try {
          await octokit.rest.git.createRef({
            owner,
            repo,
            ref: 'refs/tags/dev',
            sha,
          });
        } catch (e) {
          // we might race with others, so assume someone else has created the
          // tag by this point.
          console.log("failed to create tag: ", JSON.stringify(e.data, null, 2));
        }
      }

      console.log("double-checking tag is correct");
      tag = await octokit.request("GET /repos/:owner/:repo/git/refs/tags/:name", { owner, repo, name });
      if (tag.data.object.sha !== sha) {
        console.log("tag: ", JSON.stringify(tag.data, null, 2));
        throw new Error("tag didn't work");
      }
    } else {
      core.info(`existing tag works`);
    }
  }

  // Delete a previous release
  try {
    core.info(`fetching release`);
    let release = await octokit.rest.repos.getReleaseByTag({ owner, repo, tag: name });
    console.log("found release: ", JSON.stringify(release.data, null, 2));
    await octokit.rest.repos.deleteRelease({
      owner,
      repo,
      release_id: release.data.id,
    });
    console.log("deleted release");
  } catch (e) {
    console.log("ERROR: ", JSON.stringify(e, null, 2));
  }

  console.log("creating a release");
  let release = await octokit.rest.repos.createRelease({
    owner,
    repo,
    tag_name: name,
    prerelease: name === 'dev',
  });

  // Delete all assets from a previous run
  for (const asset of release.data.assets) {
    console.log(`deleting prior asset ${asset.id}`);
    await octokit.rest.repos.deleteReleaseAsset({
      owner,
      repo,
      asset_id: asset.id,
    });
  }

  // Upload all the relevant assets for this release as just general blobs.
  for (const file of glob.sync(files)) {
    const size = fs.statSync(file).size;
    const name = path.basename(file);
    core.info(`upload ${file}`);
    await octokit.rest.repos.uploadReleaseAsset({
      data: fs.createReadStream(file),
      headers: { 'content-length': size, 'content-type': 'application/octet-stream' },
      name,
      url: release.data.upload_url,
    });
  }
}

async function run() {
  const retries = 10;
  for (let i = 0; i < retries; i++) {
    try {
      await runOnce();
      break;
    } catch (e) {
      if (i === retries - 1)
        throw e;
      logError(e);
      console.log("RETRYING after 10s");
      await sleep(10000)
    }
  }
}

function logError(e) {
  console.log("ERROR: ", e.message);
  try {
    console.log(JSON.stringify(e, null, 2));
  } catch (e) {
    // ignore json errors for now
  }
  console.log(e.stack);
}

run().catch(err => {
  logError(err);
  core.setFailed(err.message);
});
