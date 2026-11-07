# Lint Groups

`rustc` has the concept of a "lint group", where you can toggle several warnings
through one name.

For example, the `nonstandard-style` lint sets `non-camel-case-types`,
`non-snake-case`, and `non-upper-case-globals` all at once. So these are
equivalent:

```bash
$ rustc -D nonstandard-style
$ rustc -D non-camel-case-types -D non-snake-case -D non-upper-case-globals
```

Here's a list of each lint group, and the lints that they are made up of:

{{groups-table}}

Additionally, there's a `bad-style` lint group that's a deprecated alias for `nonstandard-style`.

Finally, you can also see the table above by invoking `rustc -W help`. This will give you the exact values for the specific
compiler you have installed.
