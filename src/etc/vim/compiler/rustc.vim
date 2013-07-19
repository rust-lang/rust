" Vim compiler file
" Compiler:         Rust Compiler
" Maintainer:       Chris Morgan <me@chrismorgan.info>
" Latest Revision:  2013 Jul 12

if exists("current_compiler")
  finish
endif
let current_compiler = "rustc"

let s:cpo_save = &cpo
set cpo&vim

if exists(":CompilerSet") != 2
	command -nargs=* CompilerSet setlocal <args>
endif

if exists("g:rustc_makeprg_no_percent") && g:rustc_makeprg_no_percent == 1
	CompilerSet makeprg=rustc
else
	CompilerSet makeprg=rustc\ \%
endif

CompilerSet errorformat=
			\%f:%l:%c:\ %t%*[^:]:\ %m,
			\%f:%l:%c:\ %*\\d:%*\\d\ %t%*[^:]:\ %m,
			\%-G%f:%l\ %s,
			\%-G%*[\ ]^,
			\%-G%*[\ ]^%*[~],
			\%-G%*[\ ]...

let &cpo = s:cpo_save
unlet s:cpo_save
