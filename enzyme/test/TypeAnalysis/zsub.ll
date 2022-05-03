; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=inp -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @inp(i64* %arrayidx) {
entry:
  %ld = load i64, i64* %arrayidx, align 8, !tbaa !2
  %zsub = sub i64 0, %ld
  %zadd = add i64 0, %zsub
  ret void
}


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: inp - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %arrayidx: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* %arrayidx, align 8, !tbaa !2: {[-1]:Integer}
; CHECK-NEXT:   %zsub = sub i64 0, %ld: {[-1]:Integer}
; CHECK-NEXT:   %zadd = add i64 0, %zsub: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
