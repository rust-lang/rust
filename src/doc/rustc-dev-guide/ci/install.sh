#!/bin/bash

set -ex

function cargo_install() {
  local name=$1
  local version=$2

  if command -v $name >/dev/null 2>&1; then
    local installed_version=`$name --version | sed -E 's/[a-zA-Z_-]+ v?//g'`
    if [ "$installed_version" == "$version" ]; then
        echo "$name $version is already installed at $(command -v $name)"
    else
        echo "Forcing install $name $version"
        cargo install $name --version $version --force
    fi
  else
    echo "Installing $name $version"
    cargo install $name --version $version
  fi
}

cargo_install mdbook 0.3.0
cargo_install mdbook-linkcheck 0.3.0
