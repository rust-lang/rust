; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -ipconstprop -deadargelim -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s

define dso_local double @_Z11matvec_realPdS_(double* nocapture readonly %mat, double* nocapture readonly %vec) #4 {
entry:
  %call = tail call noalias i8* @malloc(i64 16000) #3
  %out = bitcast i8* %call to double*
  br label %for.body

for.body:                                         ; preds = %for.cond.cleanup3, %entry
  %i = phi i64 [ 0, %entry ], [ %inext, %for.cond.cleanup3 ]
  %inext = add nuw nsw i64 %i, 1
  %arrayidx = getelementptr inbounds double, double* %out, i64 %i
  store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !8
  %i2000 = mul nuw nsw i64 %i, 2000
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %for.body
  %indvars.iv54 = phi i64 [ 0, %for.body ], [ %indvars.iv.next55, %for.body4 ]
  %a2 = phi double [ 0.000000e+00, %for.body ], [ %add12, %for.body4 ]
  %a3 = add nuw nsw i64 %indvars.iv54, %i2000
  %arrayidx6 = getelementptr inbounds double, double* %mat, i64 %a3
  %a4 = load double, double* %arrayidx6, align 8, !tbaa !8
  %arrayidx8 = getelementptr inbounds double, double* %vec, i64 %indvars.iv54
  %a5 = load double, double* %arrayidx8, align 8, !tbaa !8
  %mul9 = fmul fast double %a5, %a4
  %add12 = fadd fast double %a2, %mul9
  %indvars.iv.next55 = add nuw nsw i64 %indvars.iv54, 1
  %exitcond57 = icmp eq i64 %indvars.iv.next55, 2000
  br i1 %exitcond57, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  store double %add12, double* %arrayidx, align 8, !tbaa !8
  %exitcond61 = icmp eq i64 %i, 2000
  br i1 %exitcond61, label %for.body20, label %for.body

for.body20:                                       ; preds = %for.cond.cleanup3, %for.body20
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body20 ], [ 0, %for.cond.cleanup3 ]
  %sum.050 = phi double [ %add26, %for.body20 ], [ 0.000000e+00, %for.cond.cleanup3 ]
  %arrayidx22 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  %a6 = load double, double* %arrayidx22, align 8, !tbaa !8
  %mul25 = fmul fast double %a6, %a6
  %add26 = fadd fast double %mul25, %sum.050
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2000
  br i1 %exitcond, label %for.cond.cleanup19, label %for.body20

for.cond.cleanup19:                               ; preds = %for.body20
  tail call void @free(i8* nonnull %call) #3
  ret double %add26
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #2

declare dso_local double @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(...)
; Function Attrs: norecurse uwtable
define void @caller(double* %a, double* %da, double* %b) {
entry:
  %call34.i = call double (...) @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(double (double*, double*)* nonnull @_Z11matvec_realPdS_, double* nonnull %a, double* nonnull %da, metadata !"diffe_const", double* nonnull %b)
  ret void
}

attributes #0 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { noinline noreturn nounwind }
attributes #10 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #11 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #12 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #13 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #14 = { argmemonly nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #15 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #16 = { builtin }
attributes #17 = { noreturn nounwind }
attributes #18 = { noreturn }
attributes #19 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTS7timeval", !4, i64 0, !4, i64 8}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !5, i64 0}
!10 = !{!11, !12, i64 0}
!11 = !{!"_ZTSN5adept8internal13GradientIndexILb1EEE", !12, i64 0}
!12 = !{!"int", !5, i64 0}
!13 = !{!14, !15, i64 16}
!14 = !{!"_ZTSN5adept5ArrayILi2EdLb1EEE", !15, i64 8, !15, i64 16, !16, i64 24, !16, i64 32}
!15 = !{!"any pointer", !5, i64 0}
!16 = !{!"_ZTSN5adept14ExpressionSizeILi2EEE", !5, i64 0}
!17 = !{!12, !12, i64 0}
!18 = !{!19, !15, i64 8}
!19 = !{!"_ZTSN5adept5ArrayILi1EdLb0EEE", !15, i64 0, !15, i64 8, !20, i64 16, !20, i64 20}
!20 = !{!"_ZTSN5adept14ExpressionSizeILi1EEE", !5, i64 0}
!21 = !{!19, !15, i64 0}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_: %agg.result"}
!24 = distinct !{!24, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_"}
!25 = !{!14, !15, i64 8}
!26 = !{!27, !23}
!27 = distinct !{!27, !28, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi: %agg.result"}
!28 = distinct !{!28, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi"}
!29 = !{!15, !15, i64 0}
!30 = !{!31, !12, i64 24}
!31 = !{!"_ZTSN5adept8internal16StackStorageOrigE", !15, i64 0, !15, i64 8, !15, i64 16, !12, i64 24, !12, i64 28, !12, i64 32, !12, i64 36}
!32 = !{!31, !12, i64 28}
!33 = !{!31, !15, i64 0}
!34 = !{!35, !12, i64 0}
!35 = !{!"_ZTSN5adept8internal9StatementE", !12, i64 0, !12, i64 4}
!36 = !{!31, !12, i64 32}
!37 = !{!35, !12, i64 4}
!38 = !{!39, !15, i64 0}
!39 = !{!"_ZTSSt12_Vector_baseIiSaIiEE", !40, i64 0}
!40 = !{!"_ZTSNSt12_Vector_baseIiSaIiEE12_Vector_implE", !15, i64 0, !15, i64 8, !15, i64 16}
!41 = !{!39, !15, i64 8}
!42 = !{!43, !47, i64 144}
!43 = !{!"_ZTSN5adept5StackE", !15, i64 40, !44, i64 48, !44, i64 72, !45, i64 96, !46, i64 120, !12, i64 128, !12, i64 132, !12, i64 136, !12, i64 140, !47, i64 144, !47, i64 145, !47, i64 146, !47, i64 147, !47, i64 148}
!44 = !{!"_ZTSSt6vectorIiSaIiEE"}
!45 = !{!"_ZTSNSt7__cxx114listIN5adept3GapESaIS2_EEE"}
!46 = !{!"_ZTSSt14_List_iteratorIN5adept3GapEE", !15, i64 0}
!47 = !{!"bool", !5, i64 0}
!48 = !{!43, !12, i64 128}
!49 = !{!43, !12, i64 136}
!50 = !{!51, !12, i64 8}
!51 = !{!"_ZTSN5adept6ActiveIdEE", !9, i64 0, !12, i64 8}
!52 = !{!53}
!53 = distinct !{!53, !54, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_: %agg.result"}
!54 = distinct !{!54, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_"}
!55 = !{!56, !53}
!56 = distinct !{!56, !57, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi: %agg.result"}
!57 = distinct !{!57, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi"}
!58 = !{!59}
!59 = distinct !{!59, !60, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_: %agg.result"}
!60 = distinct !{!60, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_"}
!61 = !{!62, !59}
!62 = distinct !{!62, !63, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi: %agg.result"}
!63 = distinct !{!63, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi"}
!64 = !{!65}
!65 = distinct !{!65, !66, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_: %agg.result"}
!66 = distinct !{!66, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_"}
!67 = !{!68, !65}
!68 = distinct !{!68, !69, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi: %agg.result"}
!69 = distinct !{!69, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi"}
!70 = !{!71}
!71 = distinct !{!71, !72, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_: %agg.result"}
!72 = distinct !{!72, !"_ZN5adept5ArrayILi2EdLb1EEclIiiEENS_8internal9enable_ifIXaaeqLi2ELi2Esr15all_scalar_intsIXLi2EET_T0_EE5valueENS_15ActiveReferenceIdEEE4typeES5_S6_"}
!73 = !{!74, !71}
!74 = distinct !{!74, !75, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi: %agg.result"}
!75 = distinct !{!75, !"_ZN5adept5ArrayILi2EdLb1EE20get_scalar_referenceILb1EEENS_8internal9enable_ifIXT_ENS_15ActiveReferenceIdEEE4typeERKi"}
!76 = !{!77, !15, i64 0}
!77 = !{!"_ZTSNSt8__detail15_List_node_baseE", !15, i64 0, !15, i64 8}
!78 = !{!77, !15, i64 8}
!79 = !{!4, !4, i64 0}
!80 = !{!43, !47, i64 146}
!81 = !{!43, !47, i64 147}
!82 = !{!43, !47, i64 148}
!83 = !{!31, !15, i64 8}
!84 = !{!31, !15, i64 16}
!85 = !{!31, !12, i64 36}
!86 = !{!87}
!87 = distinct !{!87, !88, !"_ZN5adept5ArrayILi1EdLb1EEclIiEENS_8internal9enable_ifIXaaaaeqLi1ELi1Esr15all_scalar_intsIXLi1EET_EE5valueLb1EENS_15ActiveReferenceIdEEE4typeES5_: %agg.result"}
!88 = distinct !{!88, !"_ZN5adept5ArrayILi1EdLb1EEclIiEENS_8internal9enable_ifIXaaaaeqLi1ELi1Esr15all_scalar_intsIXLi1EET_EE5valueLb1EENS_15ActiveReferenceIdEEE4typeES5_"}
!89 = !{!90, !15, i64 8}
!90 = !{!"_ZTSN5adept5ArrayILi1EdLb1EEE", !15, i64 8, !15, i64 16, !20, i64 24, !20, i64 28}
!91 = !{!92, !12, i64 8}
!92 = !{!"_ZTSN5adept15ActiveReferenceIdEE", !15, i64 0, !12, i64 8}
!93 = !{!94}
!94 = distinct !{!94, !95, !"_ZN5adept5ArrayILi1EdLb1EEclIiEENS_8internal9enable_ifIXaaaaeqLi1ELi1Esr15all_scalar_intsIXLi1EET_EE5valueLb1EENS_15ActiveReferenceIdEEE4typeES5_: %agg.result"}
!95 = distinct !{!95, !"_ZN5adept5ArrayILi1EdLb1EEclIiEENS_8internal9enable_ifIXaaaaeqLi1ELi1Esr15all_scalar_intsIXLi1EET_EE5valueLb1EENS_15ActiveReferenceIdEEE4typeES5_"}
!96 = !{!97}
!97 = distinct !{!97, !98, !"_ZN5adeptplINS_6ActiveIdEENS_8internal15BinaryOperationIdNS_15ActiveReferenceIdEENS3_8MultiplyES6_EEEENS3_9enable_ifIXsr8internal15rank_compatibleIXsrT_4rankEXsrT0_4rankEEE5valueENS4_INS3_7promoteINSA_4typeENSB_4typeEE4typeESA_NS3_3AddESB_EEE4typeERKNS_10ExpressionISD_SA_EERKNSL_ISE_SB_EE: %agg.result"}
!98 = distinct !{!98, !"_ZN5adeptplINS_6ActiveIdEENS_8internal15BinaryOperationIdNS_15ActiveReferenceIdEENS3_8MultiplyES6_EEEENS3_9enable_ifIXsr8internal15rank_compatibleIXsrT_4rankEXsrT0_4rankEEE5valueENS4_INS3_7promoteINSA_4typeENSB_4typeEE4typeESA_NS3_3AddESB_EEE4typeERKNS_10ExpressionISD_SA_EERKNSL_ISE_SB_EE"}
!99 = !{!90, !15, i64 16}
!100 = !{!43, !12, i64 140}
!101 = !{!102, !12, i64 4}
!102 = !{!"_ZTSN5adept3GapE", !12, i64 0, !12, i64 4}
!103 = !{!102, !12, i64 0}
!104 = !{!46, !15, i64 0}
!105 = !{!106, !15, i64 0}
!106 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !15, i64 0}
!107 = !{!108}
!108 = distinct !{!108, !109, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!109 = distinct !{!109, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!110 = !{!111, !15, i64 0}
!111 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !106, i64 0, !4, i64 8, !5, i64 16}
!112 = !{!5, !5, i64 0}
!113 = !{!111, !4, i64 8}
!114 = !{!47, !47, i64 0}
!115 = !{i8 0, i8 2}
!116 = !{!117, !15, i64 0}
!117 = !{!"_ZTSN5adept7StorageIdEE", !15, i64 0, !12, i64 8, !12, i64 12, !12, i64 16}
!118 = !{!117, !12, i64 16}
!119 = !{!117, !12, i64 12}
!120 = !{!121}
!121 = distinct !{!121, !122, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!122 = distinct !{!122, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!123 = !{!124, !124, i64 0}
!124 = !{!"vtable pointer", !6, i64 0}
!125 = !{!126, !128, i64 32}
!126 = !{!"_ZTSSt8ios_base", !4, i64 8, !4, i64 16, !127, i64 24, !128, i64 28, !128, i64 32, !15, i64 40, !129, i64 48, !5, i64 64, !12, i64 192, !15, i64 200, !130, i64 208}
!127 = !{!"_ZTSSt13_Ios_Fmtflags", !5, i64 0}
!128 = !{!"_ZTSSt12_Ios_Iostate", !5, i64 0}
!129 = !{!"_ZTSNSt8ios_base6_WordsE", !15, i64 0, !4, i64 8}
!130 = !{!"_ZTSSt6locale", !15, i64 0}
!131 = !{!132, !4, i64 8}
!132 = !{!"_ZTSSi", !4, i64 8}
!133 = !{!117, !12, i64 8}
!134 = !{!135}
!135 = distinct !{!135, !136, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!136 = distinct !{!136, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!137 = !{!51, !9, i64 0}
!138 = !{!139}
!139 = distinct !{!139, !140, !"_ZN5adept8internal13promote_arrayIdLi2EdLb1EEENS_5ArrayIXT0_ET_XT2_EEERKNS2_IXT0_ET1_XT2_EEE: %agg.result"}
!140 = distinct !{!140, !"_ZN5adept8internal13promote_arrayIdLi2EdLb1EEENS_5ArrayIXT0_ET_XT2_EEERKNS2_IXT0_ET1_XT2_EEE"}
!141 = !{!142}
!142 = distinct !{!142, !143, !"_ZN5adept8internal13promote_arrayIdLi1EdLb0EEENS_5ArrayIXT0_ET_XT2_EEERKNS2_IXT0_ET1_XT2_EEE: %agg.result"}
!143 = distinct !{!143, !"_ZN5adept8internal13promote_arrayIdLi1EdLb0EEENS_5ArrayIXT0_ET_XT2_EEERKNS2_IXT0_ET1_XT2_EEE"}
!144 = !{!145}
!145 = distinct !{!145, !146, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!146 = distinct !{!146, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!147 = !{!148}
!148 = distinct !{!148, !149, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!149 = distinct !{!149, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!150 = !{!151}
!151 = distinct !{!151, !152, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_: %agg.result"}
!152 = distinct !{!152, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_"}
!153 = !{!154}
!154 = distinct !{!154, !155, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_: %agg.result"}
!155 = distinct !{!155, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_"}
!156 = !{!157}
!157 = distinct !{!157, !158, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_S9_: %agg.result"}
!158 = distinct !{!158, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_S9_"}
!159 = !{!160}
!160 = distinct !{!160, !161, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_OS8_: %agg.result"}
!161 = distinct !{!161, !"_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_OS8_"}
!162 = !{!163, !15, i64 0}
!163 = !{!"_ZTSN5adept8internal15BinaryOperationIdNS_6ActiveIdEENS0_3AddENS1_IdNS_15ActiveReferenceIdEENS0_8MultiplyES6_EEEE", !15, i64 0, !164, i64 8}
!164 = !{!"_ZTSN5adept8internal15BinaryOperationIdNS_15ActiveReferenceIdEENS0_8MultiplyES3_EE", !15, i64 0, !15, i64 8}
!165 = !{!164, !15, i64 0}
!166 = !{!92, !15, i64 0}
!167 = !{!164, !15, i64 8}
!168 = !{!43, !15, i64 40}

; CHECK: define internal void @diffe_Z11matvec_realPdS_(double* nocapture readonly %mat, double* nocapture %"mat'", double* nocapture readonly %vec)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call noalias nonnull dereferenceable(16000) dereferenceable_or_null(16000) i8* @malloc(i64 16000)
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull dereferenceable(16000) dereferenceable_or_null(16000) i8* @malloc(i64 16000)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(16000) dereferenceable_or_null(16000) %"call'mi", i8 0, i64 16000, i1 false)
; CHECK-NEXT:   %[[outipc:.+]] = bitcast i8* %"call'mi" to double*
; CHECK-NEXT:   %out = bitcast i8* %call to double*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.cleanup3, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup3 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %out, i64 %iv
; CHECK-NEXT:   store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %i2000 = mul nuw nsw i64 %iv, 2000
; CHECK-NEXT:   br label %for.body4

; CHECK: for.body4:                                        ; preds = %for.body4, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body4 ], [ 0, %for.body ]
; CHECK-NEXT:   %a2 = phi double [ 0.000000e+00, %for.body ], [ %add12, %for.body4 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %a3 = add nuw nsw i64 %iv1, %i2000
; CHECK-NEXT:   %arrayidx6 = getelementptr inbounds double, double* %mat, i64 %a3
; CHECK-NEXT:   %a4 = load double, double* %arrayidx6, align 8, !tbaa !2
; CHECK-NEXT:   %arrayidx8 = getelementptr inbounds double, double* %vec, i64 %iv1
; CHECK-NEXT:   %a5 = load double, double* %arrayidx8, align 8, !tbaa !2
; CHECK-NEXT:   %mul9 = fmul fast double %a5, %a4
; CHECK-NEXT:   %add12 = fadd fast double %a2, %mul9
; CHECK-NEXT:   %exitcond57 = icmp eq i64 %iv.next2, 2000
; CHECK-NEXT:   br i1 %exitcond57, label %for.cond.cleanup3, label %for.body4

; CHECK: for.cond.cleanup3:                                ; preds = %for.body4
; CHECK-NEXT:   store double %add12, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond61 = icmp eq i64 %iv, 2000
; CHECK-NEXT:   br i1 %exitcond61, label %invertfor.body20, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body4
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds double, double* %[[outipc:.+]], i64 %"iv'ac.0"
; CHECK-NEXT:   store double 0.000000e+00, double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %0 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %0, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %1 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup3

; CHECK: invertfor.body4:                                  ; preds = %invertfor.cond.cleanup3, %incinvertfor.body4
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 1999, %invertfor.cond.cleanup3 ], [ %[[isub:.+]], %incinvertfor.body4 ]
; CHECK-NEXT:   %arrayidx8_unwrap = getelementptr inbounds double, double* %vec, i64 %"iv1'ac.0"
; CHECK-NEXT:   %a5_unwrap = load double, double* %arrayidx8_unwrap
; CHECK-NEXT:   %m1diffea4 = fmul fast double %[[add12de:.+]], %a5_unwrap
; CHECK-NEXT:   %i2000_unwrap = mul nuw nsw i64 %"iv'ac.0", 2000
; CHECK-NEXT:   %a3_unwrap = add nuw nsw i64 %"iv1'ac.0", %i2000_unwrap
; CHECK-NEXT:   %[[arrayidx6ipg:.+]] = getelementptr inbounds double, double* %"mat'", i64 %a3_unwrap
; CHECK-NEXT:   %[[l8:.+]] = load double, double* %[[arrayidx6ipg]], align 8
; CHECK-NEXT:   %[[addl8:.+]] = fadd fast double %[[l8]], %m1diffea4
; CHECK-NEXT:   store double %[[addl8]], double* %[[arrayidx6ipg]], align 8
; CHECK-NEXT:   %[[lcmp:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[lcmp]], label %invertfor.body, label %incinvertfor.body4

; CHECK: incinvertfor.body4:                               ; preds = %invertfor.body4
; CHECK-NEXT:   %[[isub]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body4

; CHECK: invertfor.cond.cleanup3:
; CHECK-NEXT:   %"add12'de.1" = phi double [ 0.000000e+00, %incinvertfor.body ], [ 0.000000e+00, %invertfor.body20 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %1, %incinvertfor.body ], [ 2000, %invertfor.body20 ]
; CHECK-NEXT:   %[[arrayidxipg6:.+]] = getelementptr inbounds double, double* %[[outipc]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[lipg8:.+]] = load double, double* %[[arrayidxipg6]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[arrayidxipg6]], align 8
; CHECK-NEXT:   %[[add12de]] = fadd fast double %"add12'de.1", %[[lipg8]]
; CHECK-NEXT:   br label %invertfor.body4

; CHECK: invertfor.body20:                                 ; preds = %for.cond.cleanup3, %incinvertfor.body20
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %[[iv3sub1:.+]], %incinvertfor.body20 ], [ 1999, %for.cond.cleanup3 ]
; CHECK-NEXT:   %arrayidx22_unwrap = getelementptr inbounds double, double* %out, i64 %"iv3'ac.0"
; CHECK-NEXT:   %a6_unwrap = load double, double* %arrayidx22_unwrap
; CHECK-NEXT:   %m0diffea6 = fmul fast double 1.000000e+00, %a6_unwrap
; CHECK-NEXT:   %m1diffea6 = fmul fast double 1.000000e+00, %a6_unwrap
; CHECK-NEXT:   %[[da6:.+]] = fadd fast double %m0diffea6, %m1diffea6
; CHECK-NEXT:   %[[arrayidx22ipg:.+]] = getelementptr inbounds double, double* %[[outipc]], i64 %"iv3'ac.0"
; CHECK-NEXT:   %[[l22:.+]] = load double, double* %[[arrayidx22ipg]], align 8
; CHECK-NEXT:   %[[addl22:.+]] = fadd fast double %[[l22]], %[[da6]]
; CHECK-NEXT:   store double %[[addl22]], double* %[[arrayidx22ipg]], align 8
; CHECK-NEXT:   %[[riv3cmp:.+]] = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   br i1 %[[riv3cmp]], label %invertfor.cond.cleanup3, label %incinvertfor.body20

; CHECK: incinvertfor.body20:                              ; preds = %invertfor.body20
; CHECK-NEXT:   %[[iv3sub1]] = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body20
; CHECK-NEXT: }
