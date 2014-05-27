" Language:     Rust
" Description:  Vim syntax file for Rust
" Maintainer:   Chris Morgan <me@chrismorgan.info>
" Maintainer:   Kevin Ballard <kevin@sb.org>
" Last Change:  May 27, 2014

if exists("b:did_ftplugin")
	finish
endif
let b:did_ftplugin = 1

let s:save_cpo = &cpo
set cpo&vim

" Variables {{{1

" The rust source code at present seems to typically omit a leader on /*!
" comments, so we'll use that as our default, but make it easy to switch.
" This does not affect indentation at all (I tested it with and without
" leader), merely whether a leader is inserted by default or not.
if exists("g:rust_bang_comment_leader") && g:rust_bang_comment_leader == 1
	" Why is the `,s0:/*,mb:\ ,ex:*/` there, you ask? I don't understand why,
	" but without it, */ gets indented one space even if there were no
	" leaders. I'm fairly sure that's a Vim bug.
	setlocal comments=s1:/*,mb:*,ex:*/,s0:/*,mb:\ ,ex:*/,:///,://!,://
else
	setlocal comments=s0:/*!,m:\ ,ex:*/,s1:/*,mb:*,ex:*/,:///,://!,://
endif
setlocal commentstring=//%s
setlocal formatoptions-=t formatoptions+=croqnl
" j was only added in 7.3.541, so stop complaints about its nonexistence
silent! setlocal formatoptions+=j

" This includeexpr isn't perfect, but it's a good start
setlocal includeexpr=substitute(v:fname,'::','/','g')

" NOT adding .rc as it's being phased out (0.7)
setlocal suffixesadd=.rs

if exists("g:ftplugin_rust_source_path")
    let &l:path=g:ftplugin_rust_source_path . ',' . &l:path
endif

if exists("g:loaded_delimitMate")
	if exists("b:delimitMate_excluded_regions")
		let b:rust_original_delimitMate_excluded_regions = b:delimitMate_excluded_regions
	endif
	let b:delimitMate_excluded_regions = delimitMate#Get("excluded_regions") . ',rustLifetimeCandidate,rustGenericLifetimeCandidate'
endif

" Motion Commands {{{1

" Bind motion commands to support hanging indents
nnoremap <silent> <buffer> [[ :call rust#Jump('n', 'Back')<CR>
nnoremap <silent> <buffer> ]] :call rust#Jump('n', 'Forward')<CR>
xnoremap <silent> <buffer> [[ :call rust#Jump('v', 'Back')<CR>
xnoremap <silent> <buffer> ]] :call rust#Jump('v', 'Forward')<CR>
onoremap <silent> <buffer> [[ :call rust#Jump('o', 'Back')<CR>
onoremap <silent> <buffer> ]] :call rust#Jump('o', 'Forward')<CR>

" Commands {{{1

" :Run will compile and run the current file. If it has unsaved changes, they
" will be saved first. If it has no path, it will be written to a temporary
" file first. The generated binary is always placed in a temporary directory,
" but run from the current directory.
"
" The arguments passed to :Run will be passed to the generated binary.
"
" If ! is specified, the arguments are given to rustc as well. A -- argument
" separates rustc args from the args passed to the binary.
"
" If g:rustc_path is defined, it is used as the path to rustc. Otherwise it is
" assumed that rustc is in $PATH.
command! -nargs=* -complete=file -bang -bar -buffer Run call rust#Run(<bang>0, [<f-args>])

" :Expand will expand the current file using --pretty.
"
" Any arguments given to :Expand will be passed to rustc. This is largely so
" you can pass various --cfg configurations.
"
" If ! is specified, the first argument will be interpreted as the --pretty
" type. Otherwise it will default to 'expanded'.
"
" If the current file has unsaved changes, it will be saved first. If it's an
" unnamed buffer, it will be written to a temporary file.
"
" If g:rustc_path is defined, it is used as the path to rustc. Otherwise it is
" assumed that rustc is in $PATH.
command! -nargs=* -complete=customlist,rust#CompleteExpand -bang -bar -buffer Expand call rust#Expand(<bang>0, [<f-args>])

" Mappings {{{1

" Bind ⌘R in MacVim to :Run
nnoremap <silent> <buffer> <D-r> :Run<CR>
" Bind ⌘⇧R in MacVim to :Run! pre-filled with the last args
nnoremap <buffer> <D-R> :Run! <C-r>=join(b:rust_last_rustc_args)<CR><C-\>erust#AppendCmdLine(' -- ' . join(b:rust_last_args))<CR>

if !exists("b:rust_last_rustc_args") || !exists("b:rust_last_args")
	let b:rust_last_rustc_args = []
	let b:rust_last_args = []
endif

" Cleanup {{{1

let b:undo_ftplugin = "
		\setlocal formatoptions< comments< commentstring< includeexpr< suffixesadd<
		\|if exists('b:rust_original_delimitMate_excluded_regions')
		  \|let b:delimitMate_excluded_regions = b:rust_original_delimitMate_excluded_regions
		  \|unlet b:rust_original_delimitMate_excluded_regions
		\|else
		  \|unlet! b:delimitMate_excluded_regions
		\|endif
		\|unlet! b:rust_last_rustc_args b:rust_last_args
		\|delcommand Run
		\|delcommand Expand
		\|nunmap <buffer> <D-r>
		\|nunmap <buffer> <D-R>
		\|nunmap <buffer> [[
		\|nunmap <buffer> ]]
		\|xunmap <buffer> [[
		\|xunmap <buffer> ]]
		\|ounmap <buffer> [[
		\|ounmap <buffer> ]]
		\"

" }}}1

let &cpo = s:save_cpo
unlet s:save_cpo

" vim: set noet sw=4 ts=4:
