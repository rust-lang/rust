The project is in its early stages: contributions are welcome and would be
**very** helpful, but the project is not _yet_ optimized for contribution.
Moreover, it is doubly experimental, so there's no guarantee that any work here
would reach production.

To get an idea of how rust-analyzer works, take a look at the [ARCHITECTURE.md](./ARCHITECTURE.md)
document.

Useful labels on the issue tracker:
  * [E-mentor](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-mentor)
    issues have links to the code in question and tests,
  * [E-easy](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy),
    [E-medium](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-medium),
    [E-hard](https://github.com/rust-analyzer/rust-analyzer/issues?q=is%3Aopen+is%3Aissue+label%3AE-hard),
    labels are *estimates* for how hard would be to write a fix.

There's no formal PR check list: everything that passes CI (we use [bors](https://bors.tech/)) is valid,
but it's a good idea to write nice commit messages, test code thoroughly, maintain consistent style, etc.
