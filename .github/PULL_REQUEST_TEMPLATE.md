Thank you for making Clippy better!

We're collecting our changelog from pull request descriptions.
If your PR only updates to the latest nightly, you can leave the
`changelog` entry as `none`. Otherwise, please write a short comment
explaining your change.

If your PR fixes an issue, you can add "fixes #issue_number" into this
PR description. This way the issue will be automatically closed when
your PR is merged.

If you added a new lint, here's a checklist for things that will be
checked during review or continuous integration.

- \[ ] Followed [lint naming conventions][lint_naming]
- \[ ] Added passing UI tests (including committed `.stderr` file)
- \[ ] `cargo test` passes locally
- \[ ] Executed `cargo dev update_lints`
- \[ ] Added lint documentation
- \[ ] Run `cargo dev fmt`

[lint_naming]: https://rust-lang.github.io/rfcs/0344-conventions-galore.html#lints

Note that you can skip the above if you are just opening a WIP PR in
order to get feedback.

Delete this line and everything above before opening your PR.

---

*Please keep the line below*
changelog: none
