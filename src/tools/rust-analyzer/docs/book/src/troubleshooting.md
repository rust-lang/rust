# Troubleshooting

First, search the [troubleshooting FAQ](faq.html). If your problem appears
there (and the proposed solution works for you), great! Otherwise, read on.

Start with looking at the rust-analyzer version. Try **rust-analyzer:
Show RA Version** in VS Code (using **Command Palette** feature
typically activated by Ctrl+Shift+P) or `rust-analyzer --version` in the
command line. If the date is more than a week ago, it’s better to update
rust-analyzer version.

The next thing to check would be panic messages in rust-analyzer’s log.
Log messages are printed to stderr, in VS Code you can see them in the
`Output > Rust Analyzer Language Server` tab of the panel. To see more
logs, set the `RA_LOG=info` environment variable, this can be done
either by setting the environment variable manually or by using
`rust-analyzer.server.extraEnv`, note that both of these approaches
require the server to be restarted.

To fully capture LSP messages between the editor and the server, run
the `rust-analyzer: Toggle LSP Logs` command and check `Output > Rust
Analyzer Language Server Trace`.

The root cause for many "nothing works" problems is that rust-analyzer
fails to understand the project structure. To debug that, first note the
`rust-analyzer` section in the status bar. If it has an error icon and
red, that’s the problem (hover will have somewhat helpful error
message). **rust-analyzer: Status** prints dependency information for
the current file. Finally, `RA_LOG=project_model=debug` enables verbose
logs during project loading.

If rust-analyzer outright crashes, try running
`rust-analyzer analysis-stats /path/to/project/directory/` on the
command line. This command type checks the whole project in batch mode
bypassing LSP machinery.

When filing issues, it is useful (but not necessary) to try to minimize
examples. An ideal bug reproduction looks like this:

```shell
$ git clone https://github.com/username/repo.git && cd repo && git switch --detach commit-hash
$ rust-analyzer --version
rust-analyzer dd12184e4 2021-05-08 dev
$ rust-analyzer analysis-stats .
💀 💀 💀
```

It is especially useful when the `repo` doesn’t use external crates or
the standard library.

If you want to go as far as to modify the source code to debug the
problem, be sure to take a look at the [contribution guide](contributing/index.html)!
