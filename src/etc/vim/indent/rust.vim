" Vim indent file

if exists("b:did_indent")
	finish
endif

let b:did_indent = 1

setlocal cindent
setlocal cinoptions=L0,(0,Ws,JN
setlocal cinkeys=0{,0},!^F,o,O
