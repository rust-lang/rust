; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; Function Attrs: nounwind
declare void @__enzyme_fwddiff(i8*, ...)

@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1

define void @derivative(i64* %from, i64* %fromp1, i64* %fromp2, i64* %fromp3, i64* %to, i64* %top1, i64* %top2, i64* %top3) {
entry:
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (i64*, i64*)* @callee to i8*), metadata !"enzyme_width", i64 3, metadata !"enzyme_dup", i64* %from, i64* %fromp1, i64* %fromp2, i64* %fromp3, metadata !"enzyme_dup", i64* %to, i64* %top1, i64* %top2, i64* %top3)
  ret void
}

define void @callee(i64* %from, i64* %to) {
entry:
  store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %to
  ret void
}


; CHECK: define internal void @fwddiffe3callee(i64* %from, [3 x i64*] %"from'", i64* %to, [3 x i64*] %"to'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x i64*] %"to'", 0
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %0, align 4
; CHECK-NEXT:   %1 = extractvalue [3 x i64*] %"to'", 1
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %1, align 4
; CHECK-NEXT:   %2 = extractvalue [3 x i64*] %"to'", 2
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %2, align 4
; CHECK-NEXT:   store i64 ptrtoint ([18 x i8]* @.str to i64), i64* %to, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }