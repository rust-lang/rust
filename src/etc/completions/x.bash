#!/usr/bin/env bash

_x_completions()
{
  if [ "${#COMP_WORDS[@]}" -le "2" ]; then
	COMPREPLY=($(compgen -W "build check clippy fix fmt test bench doc clean dist install run setup" "${COMP_WORDS[1]}"))
    return
  else
    compopt -o nospace
    local cur=${COMP_WORDS[COMP_CWORD]}
    local top=$(git rev-parse --show-toplevel 2>/dev/null || return)
    COMPREPLY=$(cd $top && compgen -f -- "$cur")
  fi
}
complete -F _x_completions x
