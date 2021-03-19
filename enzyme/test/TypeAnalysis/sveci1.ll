; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(<4 x i1> %arg, i64 %idx) {
entry:
  %ai = alloca <4 x i1>, align 4
  %r = shufflevector <4 x i1> %arg, <4 x i1> undef, <4 x i32> zeroinitializer
  %lai = load <4 x i1>, <4 x i1>* %ai, align 4
  %m = insertelement <4 x i1> %lai, i1 true, i64 %idx
  ret void
}

; CHECK: callee - {} |{[-1]:Integer}:{}
; CHECK-NEXT: <4 x i1> %arg: {[-1]:Integer}
; CHECK-NEXT: i64 %idx: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ai = alloca <4 x i1>, align 4: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   %r = shufflevector <4 x i1> %arg, <4 x i1> undef, <4 x i32> zeroinitializer: {[-1]:Integer}
; CHECK-NEXT:   %lai = load <4 x i1>, <4 x i1>* %ai, align 4: {[-1]:Integer}
; CHECK-NEXT:   %m = insertelement <4 x i1> %lai, i1 true, i64 %idx: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}