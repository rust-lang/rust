" Author: Kevin Ballard
" Description: Helper functions for Rust commands/mappings
" Last Modified: May 27, 2014

" Jump {{{1

function! rust#Jump(mode, function) range
	let cnt = v:count1
	normal! m'
	if a:mode ==# 'v'
		norm! gv
	endif
	let foldenable = &foldenable
	set nofoldenable
	while cnt > 0
		execute "call <SID>Jump_" . a:function . "()"
		let cnt = cnt - 1
	endwhile
	let &foldenable = foldenable
endfunction

function! s:Jump_Back()
	call search('{', 'b')
	keepjumps normal! w99[{
endfunction

function! s:Jump_Forward()
	normal! j0
	call search('{', 'b')
	keepjumps normal! w99[{%
	call search('{')
endfunction

" Run {{{1

function! rust#Run(bang, args)
	if a:bang
		let idx = index(a:args, '--')
		if idx != -1
			let rustc_args = idx == 0 ? [] : a:args[:idx-1]
			let args = a:args[idx+1:]
		else
			let rustc_args = a:args
			let args = []
		endif
	else
		let rustc_args = []
		let args = a:args
	endif

	let b:rust_last_rustc_args = rustc_args
	let b:rust_last_args = args

	call s:WithPath(function("s:Run"), rustc_args, args)
endfunction

function! s:Run(path, rustc_args, args)
	try
		let exepath = tempname()
		if has('win32')
			let exepath .= '.exe'
		endif

		let rustc_args = [a:path, '-o', exepath] + a:rustc_args

		let rustc = exists("g:rustc_path") ? g:rustc_path : "rustc"

		let output = system(shellescape(rustc) . " " . join(map(rustc_args, 'shellescape(v:val)')))
		if output != ''
			echohl WarningMsg
			echo output
			echohl None
		endif
		if !v:shell_error
			exe '!' . shellescape(exepath) . " " . join(map(a:args, 'shellescape(v:val)'))
		endif
	finally
		if exists("exepath")
			silent! call delete(exepath)
		endif
	endtry
endfunction

" Expand {{{1

function! rust#Expand(bang, args)
	if a:bang && !empty(a:args)
		let pretty = a:args[0]
		let args = a:args[1:]
	else
		let pretty = "expanded"
		let args = a:args
	endif
	call s:WithPath(function("s:Expand"), pretty, args)
endfunction

function! s:Expand(path, pretty, args)
	try
		let rustc = exists("g:rustc_path") ? g:rustc_path : "rustc"

		let args = [a:path, '--pretty', a:pretty] + a:args
		let output = system(shellescape(rustc) . " " . join(map(args, "shellescape(v:val)")))
		if v:shell_error
			echohl WarningMsg
			echo output
			echohl None
		else
			new
			silent put =output
			1
			d
			setl filetype=rust
			setl buftype=nofile
			setl bufhidden=hide
			setl noswapfile
		endif
	endtry
endfunction

function! rust#CompleteExpand(lead, line, pos)
	if a:line[: a:pos-1] =~ '^RustExpand!\s*\S*$'
		" first argument and it has a !
		let list = ["normal", "expanded", "typed", "expanded,identified", "flowgraph="]
		if !empty(a:lead)
			call filter(list, "v:val[:len(a:lead)-1] == a:lead")
		endif
		return list
	endif

	return glob(escape(a:lead, "*?[") . '*', 0, 1)
endfunction

" Emit {{{1

function! rust#Emit(type, args)
	call s:WithPath(function("s:Emit"), a:type, a:args)
endfunction

function! s:Emit(path, type, args)
	try
		let rustc = exists("g:rustc_path") ? g:rustc_path : "rustc"

		let args = [a:path, '--emit', a:type, '-o', '-'] + a:args
		let output = system(shellescape(rustc) . " " . join(map(args, "shellescape(v:val)")))
		if v:shell_error
			echohl WarningMsg
			echo output
			echohl None
		else
			new
			silent put =output
			1
			d
			if a:type == "ir"
				setl filetype=llvm
			elseif a:type == "asm"
				setl filetype=asm
			endif
			setl buftype=nofile
			setl bufhidden=hide
			setl noswapfile
		endif
	endtry
endfunction

" Utility functions {{{1

function! s:WithPath(func, ...)
	try
		let save_write = &write
		set write
		let path = expand('%')
		let pathisempty = empty(path)
		if pathisempty || !save_write
			" use a temporary file named 'unnamed.rs' inside a temporary
			" directory. This produces better error messages
			let tmpdir = tempname()
			call mkdir(tmpdir)

			let save_cwd = getcwd()
			silent exe 'lcd' tmpdir

			let path = 'unnamed.rs'

			let save_mod = &mod
			set nomod

			silent exe 'keepalt write! ' . path
			if pathisempty
				silent keepalt 0file
			endif
		else
			update
		endif

		call call(a:func, [path] + a:000)
	finally
		if exists("save_mod")   | let &mod = save_mod          | endif
		if exists("save_write") | let &write = save_write      | endif
		if exists("save_cwd")   | silent exe 'lcd' save_cwd    | endif
		if exists("tmpdir")     | silent call s:RmDir(tmpdir)  | endif
	endtry
endfunction

function! rust#AppendCmdLine(text)
	call setcmdpos(getcmdpos())
	let cmd = getcmdline() . a:text
	return cmd
endfunction

function! s:RmDir(path)
	" sanity check; make sure it's not empty, /, or $HOME
	if empty(a:path)
		echoerr 'Attempted to delete empty path'
		return 0
	elseif a:path == '/' || a:path == $HOME
		echoerr 'Attempted to delete protected path: ' . a:path
		return 0
	endif
	silent exe "!rm -rf " . shellescape(a:path)
endfunction

" }}}1

" vim: set noet sw=4 ts=4:
