; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=err -o /dev/null | FileCheck %s

define internal void @err() {
bb:
  %a3 = alloca <4 x i32>, align 4
  %wide.load.le = load <4 x i32>, <4 x i32>* %a3, align 4
  %a4 = icmp slt <4 x i32> %wide.load.le, <i32 1, i32 1, i32 1, i32 1>
  %shift = shufflevector <4 x i1> %a4, <4 x i1> zeroinitializer, <4 x i32> <i32 undef, i32 undef, i32 3, i32 undef>
  ret void
}

; CHECK: err - {} |
; CHECK-NEXT: bb
; CHECK-NEXT:   %a3 = alloca <4 x i32>, align 4: {[-1]:Pointer}
; CHECK-NEXT:   %wide.load.le = load <4 x i32>, <4 x i32>* %a3, align 4: {}
; CHECK-NEXT:   %a4 = icmp slt <4 x i32> %wide.load.le, <i32 1, i32 1, i32 1, i32 1>: {[-1]:Integer}
; CHECK-NEXT:   %shift = shufflevector <4 x i1> %a4, <4 x i1> zeroinitializer, <4 x i32> <i32 undef, i32 undef, i32 3, i32 undef>: {[-1]:Anything}
; CHECK-NEXT:   ret void: {}
