# GitHub Actions

On the GitHub hosted runners, Clippy from the latest stable Rust version comes
pre-installed. So all you have to do is to run `cargo clippy`.

```yml
on: push
name: Clippy check
jobs:
  clippy_check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v1
      - name: Run Clippy
        run: cargo clippy
```
