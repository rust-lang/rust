" Vim syntastic plugin helper
" Language:     Rust
" Maintainer:   Andrew Gallant <jamslam@gmail.com>

if exists("g:loaded_syntastic_rust_filetype")
  finish
endif
let g:loaded_syntastic_rust_filetype = 1
let s:save_cpo = &cpo
set cpo&vim

" This is to let Syntastic know about the Rust filetype.
" It enables tab completion for the 'SyntasticInfo' command.
" (This does not actually register the syntax checker.)
if exists('g:syntastic_extra_filetypes')
    call add(g:syntastic_extra_filetypes, 'rust')
else
    let g:syntastic_extra_filetypes = ['rust']
endif

let &cpo = s:save_cpo
unlet s:save_cpo
