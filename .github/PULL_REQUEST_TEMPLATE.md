Thank you for making Clippy better!

We're collecting our changelog from pull request descriptions.
If your PR only includes internal changes, you can just write
`changelog: none`. Otherwise, please write a short comment
explaining your change.

It's also helpful for us that the lint name is put within backticks (`` ` ` ``),
and then encapsulated by square brackets (`[]`), for example:
```
changelog: [`lint_name`]: your change
```

If your PR fixes an issue, you can add `fixes #issue_number` into this
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

Note that we are currently not taking in new PRs that add new lints. We are in a
feature freeze. Check out the book for more information. If you open a
feature-adding pull request, its review will be delayed.

---

*Please write a short comment explaining your change (or "none" for internal only changes)*

changelog:
