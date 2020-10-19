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
  const name = core.getInput('name');
  const token = core.getInput('token');
  const slug = process.env.GITHUB_REPOSITORY;
  const owner = slug.split('/')[0];
  const repo = slug.split('/')[1];
  const sha = process.env.HEAD_SHA;

  core.info(`files: ${files}`);
  core.info(`name: ${name}`);
  core.info(`token: ${token}`);

  const octokit = new github.GitHub(token);

  // Delete the previous release since we can't overwrite one. This may happen
  // due to retrying an upload or it may happen because we're doing the dev
  // release.
  const releases = await octokit.paginate("GET /repos/:owner/:repo/releases", { owner, repo });
  for (const release of releases) {
    if (release.tag_name !== name) {
      continue;
    }
    const release_id = release.id;
    core.info(`deleting release ${release_id}`);
    await octokit.repos.deleteRelease({ owner, repo, release_id });
  }

  // We also need to update the `dev` tag while we're at it on the `dev` branch.
  if (name == 'nightly') {
    try {
      core.info(`updating nightly tag`);
      await octokit.git.updateRef({
          owner,
          repo,
          ref: 'tags/nightly',
          sha,
          force: true,
      });
    } catch (e) {
      console.log("ERROR: ", JSON.stringify(e, null, 2));
      core.info(`creating nightly tag`);
      await octokit.git.createTag({
        owner,
        repo,
        tag: 'nightly',
        message: 'nightly release',
        object: sha,
        type: 'commit',
      });
    }
  }

  // Creates an official GitHub release for this `tag`, and if this is `dev`
  // then we know that from the previous block this should be a fresh release.
  core.info(`creating a release`);
  const release = await octokit.repos.createRelease({
    owner,
    repo,
    name,
    tag_name: name,
    target_commitish: sha,
    prerelease: name === 'nightly',
  });

  // Upload all the relevant assets for this release as just general blobs.
  for (const file of glob.sync(files)) {
    const size = fs.statSync(file).size;
    core.info(`upload ${file}`);
    await octokit.repos.uploadReleaseAsset({
      data: fs.createReadStream(file),
      headers: { 'content-length': size, 'content-type': 'application/octet-stream' },
      name: path.basename(file),
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
