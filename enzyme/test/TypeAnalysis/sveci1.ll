; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(<4 x i1> %arg) {
entry:
  %r = shufflevector <4 x i1> %arg, <4 x i1> undef, <4 x i32> zeroinitializer
  ret void
}

; CHECK: callee - {} |{[-1]:Integer}:{}
; CHECK-NEXT: <4 x i1> %arg: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r = shufflevector <4 x i1> %arg, <4 x i1> undef, <4 x i32> zeroinitializer: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
