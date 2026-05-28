# Backport Changes

Sometimes it is necessary to backport changes to the beta release of Clippy.
Backports in Clippy are rare and should be approved by the Clippy team. For
example, a backport is done, if a crucial ICE was fixed or a lint is broken to a
point, that it has to be disabled, before landing on stable.

> Note: If you think a PR should be backported you can label it with
> `beta-nominated`. This has to be done before the Thursday the week before the
> release.

## Filtering PRs to backport

First, find all labeled PRs using [this filter][beta-accepted-prs].

Next, look at each PR individually. There are a few things to check. Those need
some explanation and are quite subjective. Good judgement is required.

1. **Is the fix worth a backport?**

   This is really subjective. An ICE fix usually is. Moving a lint to a _lower_
   group (from warn- to allow-by-default) usually as well. An FP fix usually not
   (on its own). If a backport is done anyway, FP fixes might also be included.
   If the PR has a lot of changes, backports must be considered more carefully.

2. **Is the problem that was fixed by the PR already in `beta`?**

   It could be that the problem that was fixed by the PR hasn't made it to the
   `beta` branch of the Rust repo yet. If that's the case, and the fix is
   already synced to the Rust repo, the fix doesn't need to be backported, as it
   will hit stable together with the commit that introduced the problem. If the
   fix PR is not synced yet, the fix PR either needs to be "backported" to the
   Rust `master` branch or to `beta` in the next backport cycle.

3. **Make sure that the fix is on `master` before porting to `beta`**

   The fix must already be synced to the Rust `master` branch. Otherwise, the
   next `beta` will be missing this fix again. If it is not yet in `master` it
   should probably not be backported. If the backport is really important, do an
   out-of-cycle sync first. However, the out-of-cycle sync should be small,
   because the changes in that sync will get right into `beta`, without being
   tested in `nightly` first.

[beta-accepted-prs]: https://github.com/rust-lang/rust-clippy/issues?q=label%3Abeta-nominated

## Preparation

> Note: All commands in this chapter will be run in the Rust clone.

Follow the instructions in [defining remotes] to define the `clippy-upstream`
remote in the Rust repository.

After that, fetch the remote with

```bash
git fetch clippy-upstream master
```

Then, switch to the `beta` branch:

```bash
git switch beta
git fetch upstream
git reset --hard upstream/beta
```

[defining remotes]: release.md#defining-remotes

## Backport the changes

When a PR is merged with the GitHub merge queue, the PR is closed with the message

> \<PR title\> (#\<PR number\>)

This commit needs to be backported. To do that, find the `<sha1>` of that commit
and run the following command in the clone of the **Rust repository**:

```bash
git cherry-pick -m 1 `<sha1>`
```

Do this for all PRs that should be backported.

## Open PR in the Rust repository

Next, open the PR for the backport. Make sure, the PR is opened towards the
`beta` branch and not the `master` branch. The PR description should look like
this:

```
[beta] Clippy backports

r? @Mark-Simulacrum

Backports:
- <Link to the Clippy PR>
- ...

<Short summary of what is backported and why>
```

Mark is from the release team and they ultimately have to merge the PR before
branching a new `beta` version. Tag them to take care of the backport. Next,
list all the backports and give a short summary what's backported and why it is
worth backporting this.

## Relabel backported PRs

When a PR is backported to Rust `beta`, label the PR with `beta-accepted`. This
will then get picked up when [writing the changelog].

[writing the changelog]: changelog_update.md#4-include-beta-accepted-prs
