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

" smartindent will be overridden by indentexpr if filetype indent is on, but
" otherwise it's better than nothing.
setlocal smartindent nocindent

setlocal tabstop=4 shiftwidth=4 expandtab

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

" See |:RustRun| for docs
command! -nargs=* -complete=file -bang -bar -buffer RustRun call rust#Run(<bang>0, [<f-args>])

" See |:RustExpand| for docs
command! -nargs=* -complete=customlist,rust#CompleteExpand -bang -bar -buffer RustExpand call rust#Expand(<bang>0, [<f-args>])

" See |:RustEmitIr| for docs
command! -nargs=* -bar -buffer RustEmitIr call rust#Emit("ir", [<f-args>])

" See |:RustEmitAsm| for docs
command! -nargs=* -bar -buffer RustEmitAsm call rust#Emit("asm", [<f-args>])

" Mappings {{{1

" Bind ⌘R in MacVim to :RustRun
nnoremap <silent> <buffer> <D-r> :RustRun<CR>
" Bind ⌘⇧R in MacVim to :RustRun! pre-filled with the last args
nnoremap <buffer> <D-R> :RustRun! <C-r>=join(b:rust_last_rustc_args)<CR><C-\>erust#AppendCmdLine(' -- ' . join(b:rust_last_args))<CR>

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
		\|delcommand RustRun
		\|delcommand RustExpand
		\|delcommand RustEmitIr
		\|delcommand RustEmitAsm
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
