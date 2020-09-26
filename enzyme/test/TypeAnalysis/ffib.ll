; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=ffib -o /dev/null | FileCheck %s

@.str = private unnamed_addr constant [21 x i8] c"ffib'(n=%d, i=1)=%f\0A\00", align 1

; Function Attrs: nounwind readnone uwtable
define dso_local double @ffib(i32 %n, double %x) #0 {
entry:
  %cmp7 = icmp slt i32 %n, 3
  br i1 %cmp7, label %return, label %if.end

if.end:                                           ; preds = %if.end, %entry
  %n.tr9 = phi i32 [ %sub1, %if.end ], [ %n, %entry ]
  %accumulator.tr8 = phi double [ %add, %if.end ], [ %x, %entry ]
  %sub = add nsw i32 %n.tr9, -1
  %call = tail call fast double @ffib(i32 %sub, double %x)
  %sub1 = add nsw i32 %n.tr9, -2
  %add = fadd fast double %call, %accumulator.tr8
  %cmp = icmp slt i32 %n.tr9, 5
  br i1 %cmp, label %return, label %if.end

return:                                           ; preds = %if.end, %entry
  %accumulator.tr.lcssa = phi double [ %x, %entry ], [ %add, %if.end ]
  ret double %accumulator.tr.lcssa
}

; CHECK: ffib - {} |{[-1]:Integer}:{} {[-1]:Float@double}:{} 
; CHECK-NEXT: i32 %n: {[-1]:Integer}
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cmp7 = icmp slt i32 %n, 3: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp7, label %return, label %if.end: {}
; CHECK-NEXT: if.end
; CHECK-NEXT:   %n.tr9 = phi i32 [ %sub1, %if.end ], [ %n, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %accumulator.tr8 = phi double [ %add, %if.end ], [ %x, %entry ]: {[-1]:Float@double}
; CHECK-NEXT:   %sub = add nsw i32 %n.tr9, -1: {[-1]:Integer}
; CHECK-NEXT:   %call = tail call fast double @ffib(i32 %sub, double %x): {[-1]:Float@double}
; CHECK-NEXT:   %sub1 = add nsw i32 %n.tr9, -2: {[-1]:Integer}
; CHECK-NEXT:   %add = fadd fast double %call, %accumulator.tr8: {[-1]:Float@double}
; CHECK-NEXT:   %cmp = icmp slt i32 %n.tr9, 5: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %return, label %if.end: {}
; CHECK-NEXT: return
; CHECK-NEXT:   %accumulator.tr.lcssa = phi double [ %x, %entry ], [ %add, %if.end ]: {[-1]:Float@double}
; CHECK-NEXT:   ret double %accumulator.tr.lcssa: {}
