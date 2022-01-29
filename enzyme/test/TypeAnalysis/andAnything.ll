; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f(i64 %x)

define void @caller() {
entry:
  %agg.tmp.sroa.2.0.insert.ext = zext i48 undef to i64 
  %agg.tmp.sroa.2.0.insert.mask = and i64 undef, 65535 
  %agg.tmp.sroa.1.0.insert.ext = zext i8 0 to i64
  %agg.tmp.sroa.2.0.insert.shift = shl i64 %agg.tmp.sroa.2.0.insert.ext, 16
  %agg.tmp.sroa.2.0.insert.insert = or i64 %agg.tmp.sroa.2.0.insert.mask, %agg.tmp.sroa.2.0.insert.shift 
  %agg.tmp.sroa.1.0.insert.shift = shl i64 %agg.tmp.sroa.1.0.insert.ext, 8
  %agg.tmp.sroa.1.0.insert.mask = and i64 %agg.tmp.sroa.2.0.insert.insert, -65281 
  %agg.tmp.sroa.1.0.insert.insert = or i64 %agg.tmp.sroa.1.0.insert.mask, %agg.tmp.sroa.1.0.insert.shift 
  %agg.tmp.sroa.0.0.insert.ext = zext i8 undef to i64
  %agg.tmp.sroa.0.0.insert.mask = and i64 %agg.tmp.sroa.1.0.insert.insert, -256
  %agg.tmp.sroa.0.0.insert.insert = or i64 %agg.tmp.sroa.0.0.insert.mask, %agg.tmp.sroa.0.0.insert.ext
  call void @f(i64 %agg.tmp.sroa.0.0.insert.insert)
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

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %agg.tmp.sroa.2.0.insert.ext = zext i48 undef to i64: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.2.0.insert.mask = and i64 undef, 65535: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.1.0.insert.ext = zext i8 0 to i64: {[-1]:Integer}
; CHECK-NEXT:   %agg.tmp.sroa.2.0.insert.shift = shl i64 %agg.tmp.sroa.2.0.insert.ext, 16: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.2.0.insert.insert = or i64 %agg.tmp.sroa.2.0.insert.mask, %agg.tmp.sroa.2.0.insert.shift: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.1.0.insert.shift = shl i64 %agg.tmp.sroa.1.0.insert.ext, 8: {[-1]:Integer}
; CHECK-NEXT:   %agg.tmp.sroa.1.0.insert.mask = and i64 %agg.tmp.sroa.2.0.insert.insert, -65281: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.1.0.insert.insert = or i64 %agg.tmp.sroa.1.0.insert.mask, %agg.tmp.sroa.1.0.insert.shift: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.0.0.insert.ext = zext i8 undef to i64: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.0.0.insert.mask = and i64 %agg.tmp.sroa.1.0.insert.insert, -256: {[-1]:Anything}
; CHECK-NEXT:   %agg.tmp.sroa.0.0.insert.insert = or i64 %agg.tmp.sroa.0.0.insert.mask, %agg.tmp.sroa.0.0.insert.ext: {[-1]:Anything}
; CHECK-NEXT:   call void @f(i64 %agg.tmp.sroa.0.0.insert.insert): {}
; CHECK-NEXT:   ret void: {}


