; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

define <2 x double> @pmax(<2 x double> %a, <2 x double> %b) {
  %r = call <2 x double> asm "maxpd $1, $0", "=x,x,0,~{dirflag},~{fpsr},~{flags}"(<2 x double> %a, <2 x double> %b)
  ret <2 x double> %r
}

declare { <2 x double>, <2 x double> } @__enzyme_autodiff(...)

define { <2 x double>, <2 x double> } @test_derivative(<2 x double> %x, <2 x double> %y) {
entry:
  %0 = tail call { <2 x double>, <2 x double> } (...) @__enzyme_autodiff(<2 x double> (<2 x double>, <2 x double>)* @pmax, <2 x double> %x, <2 x double> %y)
  ret { <2 x double>, <2 x double> } %0
}

; CHECK: define internal { <2 x double>, <2 x double> } @diffepmax(<2 x double> %a, <2 x double> %b, <2 x double> %differeturn) 
; CHECK:   %r = call <2 x double> asm "maxpd $1, $0", "=x,x,0,~{dirflag},~{fpsr},~{flags}"(<2 x double> %a, <2 x double> %b) 
; CHECK-NEXT:   %[[i0:.+]] = fcmp fast olt <2 x double> %a, %b
; CHECK-NEXT:   %[[i1:.+]] = select {{(fast )?}}<2 x i1> %[[i0]], <2 x double> zeroinitializer, <2 x double> %differeturn
; CHECK-NEXT:   %[[i2:.+]] = select {{(fast )?}}<2 x i1> %[[i0]], <2 x double> %differeturn, <2 x double> zeroinitializer
; CHECK-NEXT:   %[[i3:.+]] = insertvalue { <2 x double>, <2 x double> } undef, <2 x double> %[[i1]], 0
; CHECK-NEXT:   %[[i4:.+]] = insertvalue { <2 x double>, <2 x double> } %[[i3]], <2 x double> %[[i2]], 1
; CHECK-NEXT:   ret { <2 x double>, <2 x double> } %[[i4]]
