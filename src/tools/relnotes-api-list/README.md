# API list generator for the release notes

Rust's [release notes] include a "Stabilized APIs" section for each
release, listing all APIs that became stable each release. This tool supports
the creation of that section by generating a JSON file containing a concise
representation of the standard library API. The [relnotes tool] will then
compare the JSON files of two releases to generate the section.

The tool is executed by CI and produces the `relnotes-api-list-$target.json`
dist artifact. You can also run the tool locally with:

```
./x dist relnotes-api-list
```

[release notes]: https://doc.rust-lang.org/stable/releases.html
[relnotes tool]: https://github.com/rust-lang/relnotes
