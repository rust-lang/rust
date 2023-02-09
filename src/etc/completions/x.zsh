#!/usr/bin/env zsh

_x_completions()
{
  local line
  local top=$(git rev-parse --show-toplevel 2>/dev/null || return)
  cd "${top}"
  _arguments -C \
    "1:subcommand:(build check clippy fix fmt test bench doc clean dist install run setup)" \
    "*:files:_files"
  cd "${OLDPWD}"
}
compdef _x_completions x
