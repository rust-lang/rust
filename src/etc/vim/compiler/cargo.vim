" Vim compiler file
" Compiler:         Cargo Compiler
" Maintainer:       Damien Radtke <damienradtke@gmail.com>
" Latest Revision:  2014 Sep 24

if exists("current_compiler")
  finish
endif
let current_compiler = "cargo"

if exists(":CompilerSet") != 2
    command -nargs=* CompilerSet setlocal <args>
endif

CompilerSet errorformat=%A%f:%l:%c:\ %m,%-Z%p^,%-C%.%#
CompilerSet makeprg=cargo\ $*

" Allow a configurable global Cargo.toml name. This makes it easy to
" support variations like 'cargo.toml'.
if !exists('g:cargo_manifest_name')
    let g:cargo_manifest_name = 'Cargo.toml'
endif

let s:local_manifest = fnamemodify(findfile(g:cargo_manifest_name, '.;'), ':p:h').'/'

if s:local_manifest != ''
    augroup cargo
        au!
        au QuickfixCmdPost make call s:FixPaths()
    augroup END

    " FixPaths() is run after Cargo, and is used to change the file paths
    " to be relative to the current directory instead of Cargo.toml.
    function! s:FixPaths()
        let qflist = getqflist()
        let manifest = s:local_manifest
        for qf in qflist
            if !qf['valid']
                let m = matchlist(qf['text'], '\v.*\(file://(.*)\)$')
                if len(m) > 0
                    let manifest = m[1].'/'
                    " Manually strip another slash if needed; usually just an
                    " issue on Windows.
                    if manifest =~ '^/[A-Z]*:/'
                        let manifest = manifest[1:]
                    endif
                endif
                continue
            endif
            let filename = bufname(qf['bufnr'])
            if filereadable(filename)
                continue
            endif
            let qf['filename'] = simplify(manifest.filename)
            call remove(qf, 'bufnr')
        endfor
        call setqflist(qflist, 'r')
    endfunction
endif
