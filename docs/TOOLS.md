# Tools used to implement libsyntax

libsyntax uses several tools to help with development. 

Each tool is a binary in the [tools/](../tools) package. 
You can run them via `cargo run` command. 

```
cargo run --package tools --bin tool
```

There are also aliases in [./cargo/config](../.cargo/config), 
so the following also works:

```
cargo tool
```


## Tool: `gen`

This tool reads a "grammar" from [grammar.ron](../grammar.ron) and
generates the `syntax_kinds.rs` file. You should run this tool if you 
add new keywords or syntax elements.


## Tool: `parse`

This tool reads rust source code from the standard input, parses it,
and prints the result to stdout.


## Tool: `collect-tests`

This tools collect inline tests from comments in libsyntax2 source code
and places them into `tests/data/inline` directory.
