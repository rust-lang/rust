; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=mv -o /dev/null | FileCheck %s

define internal void @mv(i64* %m_dims) #1 {
entry:
  %call4 = call i64 @sub(i64* nonnull %m_dims)
  ret void
}

define i64 @sub(i64* %this) {
entry:
  %agg = load i64, i64* %this
  %call = tail call i64 @pop(i64 %agg)
  ret i64 %call
}

define i64 @pop(i64 %arr.coerce0)  {
entry:
  %arr = alloca i64
  store i64 %arr.coerce0, i64* %arr
  %call.i = call i64* @cast(i64* nonnull %arr)
  %a2 = load i64, i64* %call.i, !tbaa !2
  %call2 = call i64 @mul(i64 %a2)
  ret i64 %call2
}

define i64* @cast(i64* %a) {
entry:
  ret i64* %a
}

define i64 @mul(i64 %a) {
entry:
  ret i64 %a
}

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}


; CHECK: mv - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i64* %m_dims: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call4 = call i64 @sub(i64* nonnull %m_dims): {[-1]:Integer}
; CHECK-NEXT:   ret void: {}

; CHECK: sub - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i64* %this: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %agg = load i64, i64* %this{{(, align 4)?}}: {[-1]:Integer}
; CHECK-NEXT:   %call = tail call i64 @pop(i64 %agg): {[-1]:Integer}
; CHECK-NEXT:   ret i64 %call: {}

; CHECK: pop - {} |{}:{}
; CHECK-NEXT: i64 %arr.coerce0: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %arr = alloca i64{{(, align 8)?}}: {[-1]:Pointer, [-1,-1]:Integer}
; CHECK-NEXT:   store i64 %arr.coerce0, i64* %arr{{(, align 4)?}}: {}
; CHECK-NEXT:   %call.i = call i64* @cast(i64* nonnull %arr): {[-1]:Pointer, [-1,-1]:Integer}
; CHECK-NEXT:   %a2 = load i64, i64* %call.i{{(, align 4)?}}, !tbaa !2: {[-1]:Integer}
; CHECK-NEXT:   %call2 = call i64 @mul(i64 %a2): {[-1]:Integer}
; CHECK-NEXT:   ret i64 %call2: {}

; CHECK: cast - {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer} |{[-1]:Pointer}:{}
; CHECK-NEXT: i64* %a: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   ret i64* %a: {}

; CHECK: mul - {} |{[-1]:Integer}:{}
; CHECK-NEXT: i64 %a: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   ret i64 %a: {}
