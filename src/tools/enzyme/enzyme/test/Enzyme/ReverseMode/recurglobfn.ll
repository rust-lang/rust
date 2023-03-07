; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@g = constant i8* bitcast (void (i8***)* @impl to i8*), align 8

declare i32 @offset()

define void @impl(i8*** %i) {
  %o = call i32 @offset()
  %g = getelementptr inbounds i8**, i8*** %i, i32 %o
  store i8** @g, i8*** %g, align 8
  ret void
}

define i8** @caller() {
  %i6 = call i8** (...) @__enzyme_virtualreverse(i8** @g)
  ret i8** %i6
}

declare i8** @__enzyme_virtualreverse(...)

; CHECK: @g_shadow = constant i8* bitcast ({ i8* (i8***, i8***)*, void (i8***, i8***, i8*)* }* @"_enzyme_reverse_impl'" to i8*), align 8
; CHECK: @g = constant i8* bitcast (void (i8***)* @impl to i8*), align 8, !enzyme_shadow !0
; CHECK: @"_enzyme_reverse_impl'" = internal constant { i8* (i8***, i8***)*, void (i8***, i8***, i8*)* } { i8* (i8***, i8***)* @augmented_impl, void (i8***, i8***, i8*)* @diffeimpl }

; CHECK: define i8** @caller() {
; CHECK-NEXT:   ret i8** @g_shadow
; CHECK-NEXT: }

; CHECK: define internal i8* @augmented_impl(i8*** %i, i8*** %"i'")
; CHECK-NEXT:   %o = call i32 @offset() 
; CHECK-NEXT:   %"g'ipg" = getelementptr inbounds i8**, i8*** %"i'", i32 %o
; CHECK-NEXT:   %g = getelementptr inbounds i8**, i8*** %i, i32 %o
; CHECK-NEXT:   store i8** @g_shadow, i8*** %"g'ipg", align 8
; CHECK-NEXT:   store i8** @g, i8*** %g, align 8
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }

; CHECK: define internal void @diffeimpl(i8*** %i, i8*** %"i'", i8* %tapeArg)
; CHECK-NEXT: invert:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: !0 = !{i8** @g_shadow}
