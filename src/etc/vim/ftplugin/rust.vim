" Vim syntax file
" Language:     Rust
" Maintainer:   Chris Morgan <me@chrismorgan.info>
" Last Change:  2014 Feb 27

if exists("b:did_ftplugin")
	finish
endif
let b:did_ftplugin = 1

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

" Bind motion commands to support hanging indents
nnoremap <silent> <buffer> [[ :call <SID>Rust_Jump('n', 'Back')<CR>
nnoremap <silent> <buffer> ]] :call <SID>Rust_Jump('n', 'Forward')<CR>
xnoremap <silent> <buffer> [[ :call <SID>Rust_Jump('v', 'Back')<CR>
xnoremap <silent> <buffer> ]] :call <SID>Rust_Jump('v', 'Forward')<CR>
onoremap <silent> <buffer> [[ :call <SID>Rust_Jump('o', 'Back')<CR>
onoremap <silent> <buffer> ]] :call <SID>Rust_Jump('o', 'Forward')<CR>

let b:undo_ftplugin = "
		\setlocal formatoptions< comments< commentstring< includeexpr< suffixesadd<
		\|if exists('b:rust_original_delimitMate_excluded_regions')
		  \|let b:delimitMate_excluded_regions = b:rust_original_delimitMate_excluded_regions
		  \|unlet b:rust_original_delimitMate_excluded_regions
		\|elseif exists('b:delimitMate_excluded_regions')
		  \|unlet b:delimitMate_excluded_regions
		\|endif
		\|nunmap <buffer> [[
		\|nunmap <buffer> ]]
		\|xunmap <buffer> [[
		\|xunmap <buffer> ]]
		\|ounmap <buffer> [[
		\|ounmap <buffer> ]]
		\"

if exists('*<SID>Rust_Jump') | finish | endif

function! <SID>Rust_Jump(mode, function) range
	let cnt = v:count1
	normal! m'
	if a:mode ==# 'v'
		norm! gv
	endif
	let foldenable = &foldenable
	set nofoldenable
	while cnt > 0
		execute "call <SID>Rust_Jump_" . a:function . "()"
		let cnt = cnt - 1
	endwhile
	let &foldenable = foldenable
endfunction

function! <SID>Rust_Jump_Back()
	call search('{', 'b')
	keepjumps normal! w99[{
endfunction

function! <SID>Rust_Jump_Forward()
	normal! j0
	call search('{', 'b')
	keepjumps normal! w99[{%
	call search('{')
endfunction
