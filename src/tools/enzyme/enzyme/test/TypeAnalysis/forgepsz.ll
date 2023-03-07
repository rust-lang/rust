; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64* @get()

define void @caller() {
entry:
  %p = call i64* @get()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 4, %entry ], [ %indvars.iv.next, %for.body ]
  %np = getelementptr i64, i64* %p, i64 %indvars.iv
  %ld = load i64, i64* %np, align 8, !tbaa !2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 100000000000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: entry
; CHECK-NEXT:   %p = call i64* @get(): {[-1]:Pointer, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double, [-1,80]:Float@double, [-1,88]:Float@double, [-1,96]:Float@double, [-1,104]:Float@double, [-1,112]:Float@double, [-1,120]:Float@double, [-1,128]:Float@double, [-1,136]:Float@double, [-1,144]:Float@double, [-1,152]:Float@double, [-1,160]:Float@double, [-1,168]:Float@double, [-1,176]:Float@double, [-1,184]:Float@double, [-1,192]:Float@double, [-1,200]:Float@double, [-1,208]:Float@double, [-1,216]:Float@double, [-1,224]:Float@double, [-1,232]:Float@double, [-1,240]:Float@double, [-1,248]:Float@double, [-1,256]:Float@double, [-1,264]:Float@double, [-1,272]:Float@double, [-1,280]:Float@double, [-1,288]:Float@double, [-1,296]:Float@double, [-1,304]:Float@double, [-1,312]:Float@double, [-1,320]:Float@double, [-1,328]:Float@double, [-1,336]:Float@double, [-1,344]:Float@double, [-1,352]:Float@double, [-1,360]:Float@double, [-1,368]:Float@double, [-1,376]:Float@double, [-1,384]:Float@double, [-1,392]:Float@double, [-1,400]:Float@double, [-1,408]:Float@double, [-1,416]:Float@double, [-1,424]:Float@double, [-1,432]:Float@double, [-1,440]:Float@double, [-1,448]:Float@double, [-1,456]:Float@double, [-1,464]:Float@double, [-1,472]:Float@double, [-1,480]:Float@double, [-1,488]:Float@double, [-1,496]:Float@double}
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %indvars.iv = phi i64 [ 4, %entry ], [ %indvars.iv.next, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %np = getelementptr i64, i64* %p, i64 %indvars.iv: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %np, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %indvars.iv.next = add nuw i64 %indvars.iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %exitcond = icmp eq i64 %indvars.iv, 100000000000: {[-1]:Integer}
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   ret void: {}
