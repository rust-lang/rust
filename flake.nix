{
  description = "Cryptographically verifiable Code REviews";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
    crane.url = "github:ipetkov/crane";
    crane.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, flake-compat, fenix, crane }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        lib = pkgs.lib;

        fenix-channel = fenix.packages.${system}.complete;

        fenix-toolchain = (fenix-channel.withComponents [
          "rustc"
          "cargo"
          "clippy"
          "rust-analysis"
          "rust-src"
          "rustfmt"
          "llvm-tools-preview"
        ]);

        craneLib = crane.lib.${system}.overrideToolchain fenix-toolchain;

        # filter source code at path `src` to include only the list of `modules`
        filterModules = modules: src:
          let
            basePath = toString src + "/";
          in
          lib.cleanSourceWith {
            filter = (path: type:
              let
                relPath = lib.removePrefix basePath (toString path);
                includePath =
                  (type == "directory" && builtins.match "^[^/]+$" relPath != null) ||
                  lib.any
                    (re: builtins.match re relPath != null)
                    ([ "Cargo.lock" "Cargo.toml" ".*/Cargo.toml" ] ++ builtins.concatLists (map (name: [ name "${name}/.*" ]) modules));
              in
              # uncomment to debug:
                # builtins.trace "${relPath}: ${lib.boolToString includePath}"
              includePath
            );
            inherit src;
          };

        # Filter only files needed to build project dependencies
        #
        # To get good build times it's vitally important to not have to
        # rebuild derivation needlessly. The way Nix caches things
        # is very simple: if any input file changed, derivation needs to
        # be rebuild.
        #
        # For this reason this filter function strips the `src` from
        # any files that are not relevant to the build.
        #
        # Lile `filterWorkspaceFiles` but doesn't even need *.rs files
        # (because they are not used for building dependencies)
        filterWorkspaceDepsBuildFiles = src: filterSrcWithRegexes [ "Cargo.lock" "Cargo.toml" ".*/Cargo.toml" ] src;

        # Filter only files relevant to building the workspace
        filterWorkspaceFiles = src: filterSrcWithRegexes [ "Cargo.lock" "Cargo.toml" ".*/Cargo.toml" ".*\.rs" ".*/rc/doc/.*\.md" ".*\.txt" ] src;

        filterSrcWithRegexes = regexes: src:
          let
            basePath = toString src + "/";
          in
          lib.cleanSourceWith {
            filter = (path: type:
              let
                relPath = lib.removePrefix basePath (toString path);
                includePath =
                  (type == "directory") ||
                  lib.any
                    (re: builtins.match re relPath != null)
                    regexes;
              in
              # uncomment to debug:
                # builtins.trace "${relPath}: ${lib.boolToString includePath}"
              includePath
            );
            inherit src;
          };

        commonArgs = {
          src = filterWorkspaceFiles ./.;

          buildInputs = with pkgs; [
            openssl
            fenix-channel.rustc
            fenix-channel.clippy
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
            perl
          ];

          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";
          CI = "true";
          HOME = "/tmp";
        };

        workspaceDeps = craneLib.buildDepsOnly (commonArgs // {
          src = filterWorkspaceDepsBuildFiles ./.;
          pname = "workspace-deps";
          buildPhaseCargoCommand = "cargo doc && cargo check --profile release --all-targets && cargo build --profile release --all-targets";
          doCheck = false;
        });

        # a function to define cargo&nix package, listing
        # all the dependencies (as dir) to help limit the
        # amount of things that need to rebuild when some
        # file change
        pkg = { name ? null, dir, port ? 8000, extraDirs ? [ ] }: rec {
          package = craneLib.buildPackage (commonArgs // {
            cargoArtifacts = workspaceDeps;

            src = filterModules ([ dir ] ++ extraDirs) ./.;

            # if needed we will check the whole workspace at once with `workspaceBuild`
            doCheck = false;
          } // lib.optionalAttrs (name != null) {
            pname = name;
            cargoExtraArgs = "--bin ${name}";
          });

          container = pkgs.dockerTools.buildLayeredImage {
            name = name;
            contents = [ package ];
            config = {
              Cmd = [
                "${package}/bin/${name}"
              ];
              ExposedPorts = {
                "${builtins.toString port}/tcp" = { };
              };
            };
          };
        };

        workspaceBuild = craneLib.cargoBuild (commonArgs // {
          pname = "workspace-build";
          cargoArtifacts = workspaceDeps;
          doCheck = false;
        });

        workspaceTest = craneLib.cargoBuild (commonArgs // {
          pname = "workspace-test";
          cargoArtifacts = workspaceBuild;
          doCheck = true;
        });

        # Note: can't use `cargoClippy` because it implies `--all-targets`, while
        # we can't build benches on stable
        # See: https://github.com/ipetkov/crane/issues/64
        workspaceClippy = craneLib.cargoBuild (commonArgs // {
          pname = "workspace-clippy";
          cargoArtifacts = workspaceBuild;

          cargoBuildCommand = "cargo clippy --profile release --no-deps --lib --bins --tests --examples --workspace -- --deny warnings";
          doInstallCargoArtifacts = false;
          doCheck = false;
        });

        workspaceDoc = craneLib.cargoBuild (commonArgs // {
          pname = "workspace-doc";
          cargoArtifacts = workspaceBuild;
          cargoBuildCommand = "env RUSTDOCFLAGS='-D rustdoc::broken_intra_doc_links' cargo doc --no-deps --document-private-items && cp -a target/doc $out";
          doCheck = false;
        });

        };
      in
      {
        packages = {

          deps = workspaceDeps;
          workspaceBuild = workspaceBuild;
          workspaceClippy = workspaceClippy;
          workspaceTest = workspaceTest;
          workspaceDoc = workspaceDoc;

        };

        # `nix develop`
        devShells = {
          default = pkgs.mkShell {
            buildInputs = commonArgs.buildInputs;
            nativeBuildInputs = commonArgs.nativeBuildInputs ++ (with pkgs;
              [
                fenix-toolchain
                fenix.packages.${system}.rust-analyzer

                pkgs.nixpkgs-fmt
                pkgs.shellcheck
                pkgs.rnix-lsp
                pkgs.nodePackages.bash-language-server
              ]);
            RUST_SRC_PATH = "${fenix-channel.rust-src}/lib/rustlib/src/rust/library";
            shellHook = ''
              # auto-install git hooks
              for hook in misc/git-hooks/* ; do ln -sf "../../$hook" "./.git/hooks/" ; done
              ${pkgs.git}/bin/git config commit.template misc/git-hooks/commit-template.txt

              # workaround https://github.com/rust-lang/cargo/issues/11020
              cargo_cmd_bins=( $(ls $HOME/.cargo/bin/cargo-{clippy,udeps,llvm-cov} 2>/dev/null) )
              if (( ''${#cargo_cmd_bins[@]} != 0 )); then
                echo "Warning: Detected binaries that might conflict with reproducible environment: ''${cargo_cmd_bins[@]}" 1>&2
                echo "Warning: Considering deleting them. See https://github.com/rust-lang/cargo/issues/11020 for details" 1>&2
              fi
            '';
          };

          # this shell is used only in CI, so it should contain minimum amount
          # of stuff to avoid building and caching things we don't need
          lint = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.rustfmt
              pkgs.nixpkgs-fmt
              pkgs.shellcheck
              pkgs.git
            ];
          };
        };
      });
}
