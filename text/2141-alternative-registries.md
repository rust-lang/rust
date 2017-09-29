- Feature Name: cargo_alternative_registries
- Start Date: 2017-09-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/2141
- Rust Issue: https://github.com/rust-lang/rust/issues/44931

# Summary
[summary]: #summary

This RFC proposes the addition of the support for alternative crates.io servers to be used
alongside the public crates.io server. This would allow users to publish crates to their own
private instance of crates.io, while still able to use the public instance of crates.io.

# Motivation
[motivation]: #motivation

Cargo currently has support for getting crates from a public server, which works well for open
source projects using Rust, however is problematic for closed source code. A workaround for this is
to use Git repositories to specify the packages, but that means that the helpful versioning and
discoverability that Cargo and crates.io provides is lost. We would like to change this such that
it is possible to have a local crates.io server which crates can be pushed to, while still making
use of the public crates.io server.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Registry definition specification
[registry-definition-specification]: #registry-definition-specification

We need a way to define what registries are valid for Cargo to pull from and publish to. For this
purpose, we propose that users would be able to define multiple registries in a [`.cargo/config`
file](http://doc.crates.io/config.html). This allows the user to specify the locations of
registries in one place, in a parent directory of all projects, rather than needing to configure
the registry location within each project's `Cargo.toml`. Once a registry has been configured with
a name, each `Cargo.toml` can use the registry name to refer to that registry.

Another benefit of using `.cargo/config` is that these files are not typically checked in to the
projects' source control. The registries might have credentials associated with them, which should
not be checked in. Separating the URLs and the use of the URLs in this way encourages good security
practices of not checking in credentials.

In order to tell Cargo about a registry other than crates.io, you can specify and name it in a
`.cargo/config` as follows, under the `registries` key:

```toml
[registries]
choose-a-name = "https://my-intranet:8080/index"
```

Instead of `choose-a-name`, place the name you'd like to use to refer to this registry in your
`Cargo.toml` files. The URL specified should contain the location of the registry index for this
registry; the registry format is specified in the [Registry Index Format Specification
section][registry-index-format-specification].

Alternatively, you can specify each registry as follows:

```toml
[registries.choose-a-name]
index = "https://my-intranet:8080/index"
```

If you need to specify authentication information such as a username or password to access a
registry's index, those should be specified in a `.cargo/credentials` file since it has more
restrictive file permissions than `.cargo/config`. Adding a username and password to
`.cargo/credentials` for a registry named `my-registry` would look like this:

```toml
[registries.my-registry]
username = "myusername"
password = "mypassword"
```

### CI

Because this system discourages checking in the registry configuration, the registry configuration
won't be immediately available to continuous integration systems like TravisCI. However, Cargo
currently supports configuring any key in `.cargo/config` using environment variables instead:

> Cargo can also be configured through environment variables in addition to the TOML syntax above.
> For each configuration key above of the form `foo.bar` the environment variable `CARGO_FOO_BAR`
> can also be used to define the value. For example the build.jobs key can also be defined by
> `CARGO_BUILD_JOBS`.

To configure TravisCI to use an alternate registry named `my-registry` for example, you can use
[Travis' encrypted environment variables feature](https://docs.travis-ci.com/user/environment-variables/#Defining-encrypted-variables-in-.travis.yml) to set:

`CARGO_REGISTRY_MY_REGISTRY_INDEX=https://my-intranet:8080/index`

## Using a dependency from another registry

*Note: this syntax will initially be implemented as an [unstable cargo
feature](https://github.com/rust-lang/cargo/pull/4433) available in nightly cargo only and
stabilized as it becomes ready.*

Once you've configured a registry (with a name, for example, `my-registry`) in `.cargo/config`, you
can specify that a dependency comes from an alternate registry by using the `registry` key:

```toml
[dependencies]
secret-crate = { version = "1.0", registry = "my-registry" }
```

## Publishing to another registry; preventing unwanted publishes

Today, Cargo allows you to add a key `publish = false` to your Cargo.toml to indicate that you do
not want to publish a crate anywhere. In order to specify that a crate should only be published to
a particular set of registries, this key will be extended to accept a list of registries that are
allowed with `cargo publish`:

```
publish = ["my-registry"]
```

If you run `cargo publish` without specifying an `--index` argument pointing to an allowed
registry, the command will fail. This prevents accidental publishes of private crates to crates.io,
for example.

Not having a `publish` key is equivalent to specifying `publish = true`, which means publishing to
crates.io is allowed. `publish = []` is equivalent to `publish = false`, meaning that publishing to
anywhere is disallowed.

## Running a minimal registry

The most minimal form of a registry that Cargo can use will consist of:

- A registry in the format specified in the [Registry index format specification
  section][registry-index-format-specification], which contains a pointer to:
- A location containing the `.crate` files for the crates in the registry.

## Running a fully-featured registry

This RFC does not attempt to standardize or specify any of crates.io's APIs, but it should be
possible to take crates.io's codebase and run it along with a registry index in order to provide
crates.io's functionality as an alternate registry.

## Crates.io

Because crates.io's purpose is to be a reliable host for open source crates, crates that have
dependencies from registries other than crates.io will be rejected at publish time. Crates.io
cannot make availability guarantees about alternate registries, so much like git dependencies
today, publishing with dependencies from other registries won't be allowed.

In crates.io's codebase, we will add a configuration option that specifies a list of approved
alternate registry locations that dependencies may use. For private registries run using
crates.io's code, this will likely include the private registry itself plus crates.io, so that
private crates are allowed to depend on open source crates. Any crates with dependencies from
registries not specified in this configuration option will be rejected at publish time.

## Interaction with existing features

This RFC is not proposing any changes to the way [source
replacement](http://doc.crates.io/source-replacement.html) and
[cargo-vendor](https://crates.io/crates/cargo-vendor) work; everything proposed here should be
compatible with those.

Mirrors will still be required to serve exactly the same files (matched checksums) as the source
they're mirroring.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Registry index format specification
[registry-index-format-specification]: #registry-index-format-specification

Cargo needs to be able to get a registry index containing metadata for all crates and their
dependencies available from an alternate registry in order to perform offline version resolution.
The registry index for crates.io is available at
[https://github.com/rust-lang/crates.io-index](https://github.com/rust-lang/crates.io-index), and
this section aims to specify the format of this registry index so that other registries can provide
their own registry index that Cargo will understand.

This is version 1 of the registry index format specification. There may be other versions of the
specification someday. Along with a new specification version will be a plan for supporting
registries using the older specification and a migration plan for registries to upgrade the
specification version their index is using.

A valid registry index meets the following criteria:

- The registry index is stored in a git repository so that Cargo can efficiently fetch incremental
  updates to the index.
- There will be a file at the top level named `config.json`. This file will be a valid JSON object
  with the following keys:

  ```json
  {
    "dl": "https://my-crates-server.com/api/v1/crates",
    "api": "https://my-crates-server.com/",
    "allowed-registries": ["https://github.com/rust-lang/crates.io-index", "https://my-intranet:8080/index"]
  }
  ```

  The `dl` key is required specifies where Cargo can download the tarballs containing the source
  files of the crates listed in the registry.

  The `api` key is optional and specifies where Cargo can find the API server that provides the
  same API functionality that crates.io does today, such as publishing and searching. Without the
  `api` key, these features will not be available. This RFC is not attempting to standardize
  crates.io's API in any way, although that could be a future enhancement.

  The `allowed-registries` key is optional and specifies the other registries that crates in this
  index are allowed to have dependencies on. The default will be nothing, which will mean only
  crates that depend on other crates in the current registry are allowed. This is currently the
  case for crates.io and will remain the case for crates.io going forward. Alternate registries
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

- For each crate in the registry, there will be a file with the name of that crate in the directory
  structure as specified above. The file will contain metadata about each version of the crate,
  with one version per line. Each line will be valid JSON with, minimally, the keys as shown. More
  keys may be added, but Cargo may ignore them. The contents of one line are pretty-printed here
  for readability.

  ```json
  {
      "name": "my_serde",
      "vers": "1.0.11",
      "deps": [
          {
              "name": "serde",
              "req": "^1.0",
              "registry": "https://github.com/rust-lang/crates.io-index",
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
    - `cksum`: a SHA256 checksum of the tarball downloaded
    - `features`: a list of the features available from this crate
    - `yanked`: whether or not this version has been yanked

  Within the `deps` list, each dependency should be listed as an item in the `deps` array with the
  following keys:

    - `name`: the name of the dependency
    - `req`: the semver version requirement string on this dependency
    - `registry`: **New to this RFC: the registry from which this crate is available**
    - `features`: a list of the features available from this crate
    - `optional`: whether this dependency is optional or not
    - `default_features`: whether the parent uses the default features of this dependency or not
    - `target`: on which target this dependency is needed
    - `kind`: can be `normal`, `build`, or `dev` to be a regular dependency, a build-time
      dependency, or a development dependency

If a dependency's registry is not specified, Cargo will assume the dependency can be located in the
current registry. By specifying the registry of a dependency in the index, cargo will have the
information it needs to fetch crate files from the registry indices involved without needing to
involve an API server.

## New command: `cargo generate-index-metadata`

Currently, the knowledge of how to create a file in the registry index format is spread between
Cargo and crates.io. This RFC proposes the addition of a Cargo command that would generate this
file locally for the current crate so that it can be added to the git repository using a mechanism
other than a server running crates.io's codebase.

## Related issues

In order to make working with multiple registries more convenient, we would also like to support:

- Adding a `cargo add-registry` command that could prompt for index URL and authentication
  information and place the right information in the right format in the right files to make setup
  for each user easier.
- [Being able to specify the API location rather than the index
  location](https://github.com/rust-lang/cargo/issues/4208), so that, for example, you could
  specify `https://host.company.com/api/cargo/private-repo` rather than
  `https://github.com/host-company/cargo-index`. We do not want to *require* specifying the API
  location, since some registries will choose not to have an API at all and only supply an index
  and a location for crate files. This would require the API to have a way to tell Cargo where the
  associated registry index is located.
- [Being able to save multiple tokens in
  `.cargo/credentials`](https://github.com/rust-lang/cargo/issues/3365), one per registry, so that
  people publishing to multiple registries don't need to log in over and over or specify tokens on
  every publish.
- Being able to specify `--registry registry-name` for all Cargo commands that currently take
  `--index`
- Being able to use a dependency under a different name. Alternate registries that are not mirrors
  should be allowed to have crates with the same name as crates in any other registry, including
  crates.io. In order to allow a crate to depend on both, say, the `http` crate from crates.io and
  the `http` crate from a private registry, at least one will need to be renamed when listed as a
  dependency in `Cargo.toml`. [RFC
  2126](https://github.com/aturon/rfcs/blob/path-clarity/text/0000-path-clarity.md#basic-changes)
  proposes this change as follows:

  > Cargo will provide a new crate key for aliasing dependencies, so that e.g. users who want to
  > use the `rand` crate but call it `random` instead can now write `random = { version = "0.3",
  > crate = "rand" }`.

- Being able to use environment variables to specify values in `.cargo/credentials` in the same way
  that you can use environment variables to specify values in `.cargo/config`
- For registries that don't require any authentication to access, such as public registries or
  registries only accessible within a firewall, we could support a shorthand where the index
  location (or API location when that is supported) is specified entirely within a crate dependency:

  ```toml
  [dependencies]
  my-crate = { version = "1.0", registry = "http://crate-mirror.org/index" }
  ```

  In order to discourage/disallow credentials checked in to `Cargo.toml`, if the URL contains a
  username or password, Cargo will deliberately remove it. If the registry is then inaccessible,
  the error message will mention that usernames and passwords in URLs in `Cargo.toml` are not
  allowed.

# Drawbacks
[drawbacks]: #drawbacks

Supporting alternative registries, and having multiple public registries, could fracture the
ecosystem. However, we feel that supporting private registries, and the Rust adoption that could
enable, outweighs the potential downsides of having multiple public registries.

# Rationale and Alternatives
[alternatives]: #alternatives

A [previous RFC](https://github.com/rust-lang/rfcs/pull/2006) proposed having the registry
information completely defined within `Cargo.toml` rather than using `.cargo/config`. This requires
repeating the same information multiple times for multiple projects, and encourages checking in
credentials that might be needed to access the registries. That RFC also didn't specify the format
for the registry index, which needs to be shared among all registries.

An alternative design could be to support specifying the registry URL in either `.cargo/config` or
`Cargo.toml`. This has the downsides of creating more choices for the user and potentially
encouraging poor practices such as checking credentials into a project's source control. The
implementation of this feature would also be more complex. The upside would be supporting
configuration in ways that would be more convenient in various situations.

# Unresolved questions
[unresolved]: #unresolved-questions

- Are the names of everything what we want?
  - `cargo generate-index-metadata`?
  - `registry = my-registry`?
  - `publish-registries = []`?

- What kinds of authentication parameters do we need to support in `.cargo/credentials`?
