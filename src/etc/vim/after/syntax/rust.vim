if !exists('g:rust_conceal') || !has('conceal') || &enc != 'utf-8'
	finish
endif

" For those who don't want to see `::`...
if exists('g:rust_conceal_mod_path')
	syn match rustNiceOperator "::" conceal cchar=ㆍ
endif

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

" For those who don't want to see `pub`...
if exists('g:rust_conceal_pub')
    syn match rustPublicSigil contained "pu" conceal cchar=＊
    syn match rustPublicRest contained "b" conceal cchar= 
    syn match rustNiceOperator "pub " contains=rustPublicSigil,rustPublicRest
endif

hi link rustNiceOperator Operator

if !exists('g:rust_conceal_mod_path')
    hi! link Conceal Operator
endif

setlocal conceallevel=2
