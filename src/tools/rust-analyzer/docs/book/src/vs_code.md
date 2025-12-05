# VS Code

This is the best supported editor at the moment. The rust-analyzer
plugin for VS Code is maintained [in
tree](https://github.com/rust-lang/rust-analyzer/tree/master/editors/code).

You can install the latest release of the plugin from [the
marketplace](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).

Note that the plugin may cause conflicts with the [previous official
Rust
plugin](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust).
The latter is no longer maintained and should be uninstalled.

The server binary is stored in the extension install directory, which
starts with `rust-lang.rust-analyzer-` and is located under:

-   Linux: `~/.vscode/extensions`

-   Linux (Remote, such as WSL): `~/.vscode-server/extensions`

-   macOS: `~/.vscode/extensions`

-   Windows: `%USERPROFILE%\.vscode\extensions`

As an exception, on NixOS, the extension makes a copy of the server and
stores it under
`~/.config/Code/User/globalStorage/rust-lang.rust-analyzer`.

Note that we only support the two most recent versions of VS Code.

### Updates

The extension will be updated automatically as new versions become
available. It will ask your permission to download the matching language
server version binary if needed.

#### Nightly

We ship nightly releases for VS Code. To help us out by testing the
newest code, you can enable pre-release versions in the Code extension
page.

### Manual installation

Alternatively, download a VSIX corresponding to your platform from the
[releases](https://github.com/rust-lang/rust-analyzer/releases) page.

Install the extension with the `Extensions: Install from VSIX` command
within VS Code, or from the command line via:

    $ code --install-extension /path/to/rust-analyzer.vsix

If you are running an unsupported platform, you can install
`rust-analyzer-no-server.vsix` and compile or obtain a server binary.
Copy the server anywhere, then add the path to your settings.json, for
example:

```json
{ "rust-analyzer.server.path": "~/.local/bin/rust-analyzer-linux" }
```

### Building From Source

Both the server and the Code plugin can be installed from source:

    $ git clone https://github.com/rust-lang/rust-analyzer.git && cd rust-analyzer
    $ cargo xtask install

You’ll need Cargo, nodejs (matching a supported version of VS Code) and
npm for this.

Note that installing via `xtask install` does not work for VS Code
Remote, instead you’ll need to install the `.vsix` manually.

If you’re not using Code, you can compile and install only the LSP
server:

    $ cargo xtask install --server

Make sure that `.cargo/bin` is in `$PATH` and precedes paths where
`rust-analyzer` may also be installed. Specifically, `rustup` includes a
proxy called `rust-analyzer`, which can cause problems if you’re
planning to use a source build or even a downloaded binary.

## VS Code or VSCodium in Flatpak

Setting up `rust-analyzer` with a Flatpak version of Code is not trivial
because of the Flatpak sandbox. While the sandbox can be disabled for
some directories, `/usr/bin` will always be mounted under
`/run/host/usr/bin`. This prevents access to the system’s C compiler, a
system-wide installation of Rust, or any other libraries you might want
to link to. Some compilers and libraries can be acquired as Flatpak
SDKs, such as `org.freedesktop.Sdk.Extension.rust-stable` or
`org.freedesktop.Sdk.Extension.llvm15`.

If you use a Flatpak SDK for Rust, it must be in your `PATH`:

 * install the SDK extensions with `flatpak install org.freedesktop.Sdk.Extension.{llvm15,rust-stable}//23.08`
 * enable SDK extensions in the editor with the environment variable `FLATPAK_ENABLE_SDK_EXT=llvm15,rust-stable` (this can be done using flatseal or `flatpak override`)

If you want to use Flatpak in combination with `rustup`, the following
steps might help:

-   both Rust and `rustup` have to be installed using
    <https://rustup.rs>. Distro packages *will not* work.

-   you need to launch Code, open a terminal and run `echo $PATH`

-   using
    [Flatseal](https://flathub.org/apps/details/com.github.tchx84.Flatseal),
    you must add an environment variable called `PATH`. Set its value to
    the output from above, appending `:~/.cargo/bin`, where `~` is the
    path to your home directory. You must replace `~`, as it won’t be
    expanded otherwise.

-   while Flatseal is open, you must enable access to "All user files"

A C compiler should already be available via `org.freedesktop.Sdk`. Any
other tools or libraries you will need to acquire from Flatpak.

