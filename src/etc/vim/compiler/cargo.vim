" Vim compiler file
" Compiler:         Cargo Compiler
" Maintainer:       Damien Radtke <damienradtke@gmail.com>
" Latest Revision:  2014 Sep 24

if exists('current_compiler')
  finish
endif
runtime compiler/rustc.vim
let current_compiler = "cargo"

if exists(':CompilerSet') != 2
    command -nargs=* CompilerSet setlocal <args>
endif

if exists('g:cargo_makeprg_params')
    execute 'CompilerSet makeprg=cargo\ '.escape(g:cargo_makeprg_params, ' \|"').'\ $*'
else
    CompilerSet makeprg=cargo\ $*
endif

" Allow a configurable global Cargo.toml name. This makes it easy to
" support variations like 'cargo.toml'.
let s:cargo_manifest_name = get(g:, 'cargo_manifest_name', 'Cargo.toml')

function! s:is_absolute(path)
    return a:path[0] == '/' || a:path =~ '[A-Z]\+:'
endfunction

let s:local_manifest = findfile(s:cargo_manifest_name, '.;')
if s:local_manifest != ''
    let s:local_manifest = fnamemodify(s:local_manifest, ':p:h').'/'
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
            if !qf.valid
                let m = matchlist(qf.text, '(file://\(.*\))$')
                if !empty(m)
                    let manifest = m[1].'/'
                    " Manually strip another slash if needed; usually just an
                    " issue on Windows.
                    if manifest =~ '^/[A-Z]\+:/'
                        let manifest = manifest[1:]
                    endif
                endif
                continue
            endif
            let filename = bufname(qf.bufnr)
            if s:is_absolute(filename)
                continue
            endif
            let qf.filename = simplify(manifest.filename)
            call remove(qf, 'bufnr')
        endfor
        call setqflist(qflist, 'r')
    endfunction
endif
