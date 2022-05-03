; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

define i64 @f() {
entry:
  %all = alloca i64, align 8
  store i64 0, i64* %all, align 8
  %res = load i64, i64* %all, align 8
  ret i64 %res
}

define void @caller(i64* %p) {
entry:
  %sub = call i64 @f()
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %next, %for.body ]
  %next = add nsw i64 %iv, %sub
  %cmp = icmp ugt i64 %iv, 10
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

; CHECK: f - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %all = alloca i64, align 8: {[-1]:Pointer}
; CHECK-NEXT:   store i64 0, i64* %all, align 8: {}
; CHECK-NEXT:   %res = load i64, i64* %all, align 8: {}
; CHECK-NEXT:   ret i64 %res: {}

; CHECK: caller - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %p: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %sub = call i64 @f(): {}
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %next, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %next = add nsw i64 %iv, %sub: {}
; CHECK-NEXT:   %cmp = icmp ugt i64 %iv, 10: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.cond.cleanup: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   ret void: {}
