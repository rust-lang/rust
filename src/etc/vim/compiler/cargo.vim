" Vim compiler file
" Compiler:         Cargo Compiler
" Maintainer:       Damien Radtke <damienradtke@gmail.com>
" Latest Revision:  2014 Sep 18

if exists("current_compiler")
  finish
endif
let current_compiler = "cargo"

if exists(":CompilerSet") != 2
    command -nargs=* CompilerSet setlocal <args>
endif

CompilerSet errorformat&
CompilerSet makeprg=cargo\ $*

" Allow a configurable global Cargo.toml name. This makes it easy to
" support variations like 'cargo.toml'.
if !exists('g:cargo_toml_name')
    let g:cargo_toml_name = 'Cargo.toml'
endif

let s:toml_dir = fnamemodify(findfile(g:cargo_toml_name, '.;'), ':p:h').'/'

if s:toml_dir != ''
    augroup cargo
        au!
        au QuickfixCmdPost make call s:FixPaths()
    augroup END

    " FixPaths() is run after Cargo, and is used to change the file paths
    " to be relative to the current directory instead of Cargo.toml.
    function! s:FixPaths()
        let qflist = getqflist()
        for qf in qflist
            if !qf['valid']
                continue
            endif
            let filename = bufname(qf['bufnr'])
            if stridx(filename, s:toml_dir) == -1
                let filename = s:toml_dir.filename
            endif
            let qf['filename'] = simplify(s:toml_dir.bufname(qf['bufnr']))
            call remove(qf, 'bufnr')
        endfor
        call setqflist(qflist, 'r')
    endfunction
endif
