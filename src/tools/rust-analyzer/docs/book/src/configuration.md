# Configuration

**Source:**
[config.rs](https://github.com/rust-lang/rust-analyzer/blob/master/crates/rust-analyzer/src/config.rs)

The [Installation](./installation.md) section contains details on
configuration for some of the editors. In general `rust-analyzer` is
configured via LSP messages, which means that it’s up to the editor to
decide on the exact format and location of configuration files.

Some clients, such as [VS Code](./vs_code.md) or [COC plugin in
Vim](./other_editors.md#coc-rust-analyzer) provide `rust-analyzer` specific configuration
UIs. Others may require you to know a bit more about the interaction
with `rust-analyzer`.

For the later category, it might help to know that the initial
configuration is specified as a value of the `initializationOptions`
field of the [`InitializeParams` message, in the LSP
protocol](https://microsoft.github.io/language-server-protocol/specifications/specification-current/#initialize).
The spec says that the field type is `any?`, but `rust-analyzer` is
looking for a JSON object that is constructed using settings from the
list below. Name of the setting, ignoring the `rust-analyzer.` prefix,
is used as a path, and value of the setting becomes the JSON property
value.

For example, a very common configuration is to enable proc-macro
support, can be achieved by sending this JSON:

    {
      "cargo": {
        "buildScripts": {
          "enable": true,
        },
      },
      "procMacro": {
        "enable": true,
      }
    }

Please consult your editor’s documentation to learn more about how to
configure [LSP
servers](https://microsoft.github.io/language-server-protocol/).

To verify which configuration is actually used by `rust-analyzer`, set
`RA_LOG` environment variable to `rust_analyzer=info` and look for
config-related messages. Logs should show both the JSON that
`rust-analyzer` sees as well as the updated config.

This is the list of config options `rust-analyzer` supports:

{{#include configuration_generated.md}}
