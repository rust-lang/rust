- Feature Name: (fill me in with a unique ident, my_awesome_feature)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes the addition of the support for alternative crates.io servers to be used
alongside the public crates.io server. This would allow users to publish crates to their
own private instance of crates.io, while still able to use the public instance of crates.io.

# Motivation
[motivation]: #motivation

Cargo currently has support for getting crates from a public server, which works well for open
source projects using Rust, however is problematic for closed source code. A workaround for
this is to use Git repositories to specify the packages, but that means that the helpful
versioning and discoverability that Cargo and crates.io provides is lost. We would like to
change this such that it is possible to have a local crates.io server which crates can be
pushed to, while still making use of the public crates.io server.

We would also like to support the use of crates.io mirrors. These differ from alternative
registries in that a mirror completely replicates the functionality and content of
crates.io. A mirror would be useful if we ever need a fallback for when crates.io
goes down, or in areas of the world where crates.io is blocked.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Registry definition specification

We need a way to define what registries are valid for Cargo to pull from and publish to. For this
purpose, we propose that users would be able to define multiple registries in the global
`~/.cargo/config` file. This would allow the user to specify what registry they want to publish
to without binding registries directly with projects.

config doesn't have to be global - goes up reference tree from project to project, goes to first
one finds

token moved to cargo credentials so different file permissions
registry file index location exists, keep
need to add name

[registry.new-registry]
index or host
[registry.other-registry]

in cargo toml could say registry = new-registry, and would know to look in config for registry host/index


TODO what do users specify in the config for the registry? name, where does the token come from?
TODO how do individual crates specify what registry to use?

Explain the proposal as if it was already included in the language and you were teaching it to another Rust programmer. That generally means:

- Introducing new named concepts.
- Explaining the feature largely in terms of examples.
- Explaining how Rust programmers should *think* about the feature, and how it should impact the way they use Rust. It should explain the impact as concretely as possible.
- If applicable, provide sample error messages, deprecation warnings, or migration guidance.
- If applicable, describe the differences between teaching this to existing Rust programmers and new Rust programmers.

For implementation-oriented RFCs (e.g. for compiler internals), this section should focus on how compiler contributors should think about the change, and give examples of its concrete impact. For policy RFCs, this section should provide an example-driven introduction to the policy, and explain its impact in concrete terms.

----

A crate that describes where it publishes to would add
the `registry` key to the `package` section of Cargo.toml:

```toml
[package]
name = "serde"
registry = "http://example.com/"
```

A crate using a dependency from a different registry would add
the `registry` key to the `dependencies` section of Cargo.toml:

```toml
[dependencies.serde]
registry = "http://example.com/"
```

Without further configuration, the value of the key for `registry`
will be used as the URL for the registry. Optionally, a user can
configure settings for a registry in the `.cargo` configuration files:

```
[registries."http://example.com/"]
url = "https://example.org/api"
username = "anna"
token = "secret"
```

This allows for user-wide settings like usernames and tokens. A user
may even use a `registry` key like `my-registry` if they wanted
increased indirection.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Registry index format specification

Cargo needs to be able to get a registry index containing metadata for all
crates and their dependencies available from an alternate registry in order to
perform offline version resolution. The registry index for crates.io is
available at
[https://github.com/rust-lang/crates.io-index](https://github.com/rust-lang/crat
es.io-index), and this section aims to specify the format of this registry
index so that other registries can provide their own registry index that Cargo
will understand.

This is version 1 of the registry index format specification. There may be
other versions of the specification someday. Along with a new specification
version will be a plan for supporting registries using the older specification
and a migration plan for registries to upgrade the specification version their
index is using.

A valid registry index meets the following criteria:

- The registry index is stored in a git repository so that Cargo can
  efficiently fetch incremental updates to the index.
- There will be a file at
  the top level named `config.json`. This file will be a valid JSON object with
  the following keys:

  ```json
  {
    "dl": "https://my-crates-server.com/api/v1/crates",
    "api": "https://my-crates-server.com/",
    "allowed-registries": ["https://crates.io", "https://my-other-crates-server.com"]
  }
  ```

  The `dl` key is required specifies where Cargo can download the tarballs
  containing the source files of the crates listed in the registry.

  The `api` key is optional and specifies where Cargo can find the API server
  that provides the same API functionality that crates.io does today, such as
  publishing and searching. Without the `api` key, these features will not be
  available. This RFC is not attempting to standardize crates.io's API in any
  way, although that could be a future enhancement.

  The `allowed-registries` key is optional and specifies the other registries
  that crates in this index are allowed to have dependencies on. The default
  will be nothing, which will mean only crates that depend on other crates in
  the current registry are allowed. This is currently the case for crates.io
  and will remain the case for crates.io going forward. Alternate registries
  will probably want to add crates.io to this list.

- There will be a number of directories in the git repository.
  - `1/` - holds files for all crates whose names have one letter.
  - `2/` - holds files for all crates whose names have two letters.
  - `3/` - holds files for all crates whose names have three letters.
  - `aa/aa/` etc - for all crates whose names have four or more letters, their
    files will be in a directory named with the first and second letters of
    their name, then in a subdirectory named with the third and fourth letters
    of their name. For example, a file for a crate named `sample` would be
    found in `sa/mp/`.

- For each crate in the registry, there will be a file with the name of that
  crate in the directory structure as specified above. The file will contain
  metadata about each version of the crate, with one version per line. Each
  line will be valid JSON with, minimally, the keys as shown. More keys may be
  added, but Cargo may ignore them. The contents of one line are pretty-printed
  here for readability.

  ```json
  {
      "name": "my_serde",
      "vers": "1.0.11",
      "deps": [
          {
              "name": "serde",
              "req": "^1.0",
              "registry": "https://crates.io",
              "features": [],
              "optional": true,
              "default_features": true,
              "target": null,
              "kind": "normal"
          }
      ],
      "cksum": "f7726f29ddf9731b17ff113c461e362c381d9d69433f79de4f3dd572488823e9",
      "features": {
          "default": [
              "std"
          ],
          "derive": [
              "serde_derive"
          ],
          "std": [

          ],
      },
      "yanked": false
  }
  ```

  The top-level keys for a crate are:

    - `name`: the name of the crate
    - `vers`: the version of the crate this row is describing
    - `deps`: a list of all dependencies of this crate
    - `cksum`: a checksum of this version's files
    - `features`: a list of the features available from this crate
    - `yanked`: whether or not this version has been yanked

  Within the `deps` list, each dependency should be listed as an item in the `deps` array with the following keys:

    - `name`: the name of the dependency
    - `req`: the semver version requirement string on this dependency
    - `registry`: **New to this RFC: the registry from which this crate is
      available**
    - `features`: a list of the features available from this crate
    - `optional`: whether this dependency is optional or not
    - `default_features`: whether the parent crate uses the default features of
      this dependency or not
    - `target`: on which target this dependency is needed
    - `kind`: can be `normal`, `build`, or `dev` to be a regular dependency, a
      build-time dependency, or a development dependency

If a dependency's registry is not specified, Cargo will assume the dependency can be located in the current registry. By specifying the registry of a dependency in the index, cargo will have the information it needs to fetch crate files from the registry indices involved without needing to involve an API server.

Currently, the knowledge of how to create a file in this format is spread
between Cargo and crates.io. This RFC proposes the addition of a Cargo command
that would generate this file locally for the current crate so that it can be
added to the git repository using a mechanism other than a server running
crates.io's codebase.

# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and Alternatives
[alternatives]: #alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Unresolved questions
[unresolved]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
