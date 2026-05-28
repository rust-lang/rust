{
  description = "rustc dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      inherit (nixpkgs) lib;
      forEachSystem = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forEachSystem (system: {
        default = nixpkgs.legacyPackages.${system}.callPackage ./shell.nix { };
      });

      packages = forEachSystem (system: {
        default = nixpkgs.legacyPackages.${system}.callPackage ./x { };
      });
    };
}
