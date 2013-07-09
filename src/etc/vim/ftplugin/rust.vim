" Vim syntax file
" Language:     Rust
" Maintainer:   Chris Morgan <me@chrismorgan.info>
" Last Change:  2013 Jul 6

if exists("b:did_ftplugin")
	finish
endif
let b:did_ftplugin = 1

setlocal comments=s1:/*,mb:*,ex:*/,:///,://!,://
setlocal commentstring=//%s
setlocal formatoptions-=t formatoptions+=croqnlj

" This includeexpr isn't perfect, but it's a good start
setlocal includeexpr=substitute(v:fname,'::','/','g')

" NOT adding .rc as it's being phased out (0.7)
setlocal suffixesadd=.rs

if exists("g:ftplugin_rust_source_path")
    let &l:path=g:ftplugin_rust_source_path . ',' . &l:path
endif

let b:undo_ftplugin = "setlocal formatoptions< comments< commentstring< includeexpr< suffixesadd<"
