# Contributing Guide

Thank you for your interest in contributing to Enzyme! There are multiple ways to contribute, and we appreciate all contributions. In case you have questions, you can either use the [Discourse](https://discourse.llvm.org/c/projects-that-want-to-become-official-llvm-projects/enzyme/45) or the #enzyme channel on [Discord](https://discord.gg/xS7Z362).


## Ways to Contribute

### Bug Reports and Feature Requests

If you are working with Enzyme, and run into a bug, or require additional features, we definitely want to know about it. Please let us know by either creating a [GitHub Issues](https://github.com/EnzymeAD/Enzyme/issues), or starting a thread in Enzyme's [LLVM Discourse Category](https://discourse.llvm.org/c/projects-that-want-to-become-official-llvm-projects/enzyme/45).

### Bug Fixes

If you are interested in contributing code to Enzyme, issues labeled with the [good-first-issue](https://github.com/EnzymeAD/Enzyme/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) keyword in the [GitHub Issues](https://github.com/EnzymeAD/Enzyme/issues) are a good way to get familiar with the code base. If you are interested in fixing a bug please comment on it to let people know you are working on it.

> The easiest way to explore code in Enzyme is by using our [Compiler Explorer](https://enzyme.mit.edu/explorer) instance.

If you are unable to reproduce the bug in Compiler Explorer, try to reproduce and fix the bug with upstream LLVM. Start by building LLVM from source as described in [Installation](https://enzyme.mit.edu/Installation/) and use the built binaries to reproduce the failure described in the bug.

### Language Integrations

In case you are interested in integrating Enzyme with a programming language which compiles to the LLVM IR, we encourage you to take a look at the [Language Frontend Guide](), and post on the [Discourse Category](https://discourse.llvm.org/c/projects-that-want-to-become-official-llvm-projects/enzyme/45) to let people know that you are working on it.

## How to Submit a Pull Request

Once you have a pull request ready, it is time to submit it. The pull request should:

- Include a small unit test
- Conform to the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html). You can use the [clang-format-diff.py](https://reviews.llvm.org/source/llvm-github/browse/main/clang/tools/clang-format/clang-format-diff.py) or [git-clang-format](https://reviews.llvm.org/source/llvm-github/browse/main/clang/tools/clang-format/git-clang-format) tools to automatically format your patch properly.
- Not contain any unrelated changes
- Be an isolated change. Independent changes should be submitted as separate pull requests as this makes reviewing easier.

Before sending a pull request for review, please also try to ensure it is formatted properly. We use clang-format for this, which has git integration through the git-clang-format script. On some systems, it may already be installed (or be installable via your package manager). If so, you can simply run it – the following command will format only the code changed in the most recent commit:

```bash
git clang-format HEAD~1
```

Note that this modifies the files, but doesn't commit them - you'll likely want to run

```bash
git commit --amend -a
```

in order to update the latest commit with all pending changes.

> If you don't already have `clang-format` or `git-clang-format` installed on your system, the `clang-format` binary will be built alongside clang, and the git integration can be run form `clang/tools/clang-format/git-clang-format`.

## Code of Conduct

As an LLVM Incubator Project we expect all participants to abide by the [LLVM Community Code of Conduct](https://llvm.org/docs/CodeOfConduct.html).
