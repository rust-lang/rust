if exists('g:no_rust_conceal') || !has('conceal') || &enc != 'utf-8'
	finish
endif

" For those who don't want to see `::`...
if exists('g:rust_conceal_mod_path')
	syn match rustNiceOperator "::" conceal cchar=ㆍ
endif

syn match rustLeftArrowHead contained "-" conceal cchar= 
syn match rustLeftArrowTail contained "<" conceal cchar=⟵
syn match rustNiceOperator "<-" contains=rustLeftArrowHead,rustLeftArrowTail

syn match rustRightArrowHead contained ">" conceal cchar= 
syn match rustRightArrowTail contained "-" conceal cchar=⟶
syn match rustNiceOperator "->" contains=rustRightArrowHead,rustRightArrowTail

syn match rustLeftRightArrowHead contained ">" conceal cchar= 
syn match rustLeftRightArrowTail contained "<-" conceal cchar=⟷
syn match rustNiceOperator "<->" contains=rustLeftRightArrowHead,rustLeftRightArrowTail

syn match rustFatRightArrowHead contained ">" conceal cchar= 
syn match rustFatRightArrowTail contained "=" conceal cchar=⟹
syn match rustNiceOperator "=>" contains=rustFatRightArrowHead,rustFatRightArrowTail

syn match rustNiceOperator /\<\@!_\(_*\>\)\@=/ conceal cchar=′

hi link rustNiceOperator Operator
hi! link Conceal Operator
setlocal conceallevel=2
