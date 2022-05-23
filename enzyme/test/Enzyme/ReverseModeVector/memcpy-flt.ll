; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

; Function Attrs: nounwind
declare void @__enzyme_autodiff.f64(...)

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(double* nocapture %dst, double* nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double* %dst to i8*
  %1 = bitcast double* %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(double* %dst, double* %dstp1, double* %dstp2, double* %dstp3, double* %src, double* %srcp1, double* %dsrcp2, double* %dsrcp3, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_width", i64 3, double* %dst, double* %dstp1, double* %dstp2, double* %dstp3, double* %src, double* %srcp1, double* %dsrcp2, double* %dsrcp3, i64 %n) #3
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }


; CHECK: define internal void @diffe3memcpy_float(double* nocapture %dst, [3 x double*] %"dst'", double* nocapture readonly %src, [3 x double*] %"src'", i64 %num)
; CHECK-NEXT:  entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"dst'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"dst'", 1
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"dst'", 2
; CHECK-NEXT:   %3 = bitcast double* %dst to i8*
; CHECK-NEXT:   %4 = extractvalue [3 x double*] %"src'", 0
; CHECK-NEXT:   %5 = extractvalue [3 x double*] %"src'", 1
; CHECK-NEXT:   %6 = extractvalue [3 x double*] %"src'", 2
; CHECK-NEXT:   %7 = bitcast double* %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %3, i8* align 1 %7, i64 %num, i1 false)
; CHECK-NEXT:   %8 = lshr i64 %num, 3
; CHECK-NEXT:   %9 = {{(icmp eq i64 %8, 0|icmp ult i64 %num, 8)}}
; CHECK-NEXT:   br i1 %9, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 1
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %4, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i, align 1
; CHECK-NEXT:   %10 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %10, double* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %11 = icmp eq i64 %8, %idx.next.i
; CHECK-NEXT:   br i1 %11, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda1sa1.exit:             ; preds = %entry, %for.body.i
; CHECK-NEXT:   %12 = lshr i64 %num, 3
; CHECK-NEXT:   %13 = {{(icmp eq i64 %12, 0|icmp ult i64 %num, 8)}}
; CHECK-NEXT:   br i1 %13, label %__enzyme_memcpyadd_doubleda1sa1.exit13, label %for.body.i12

; CHECK: or.body.i12:                                     ; preds = %for.body.i12, %__enzyme_memcpyadd_doubleda1sa1.exit
; CHECK-NEXT:  %idx.i6 = phi i64 [ 0, %__enzyme_memcpyadd_doubleda1sa1.exit ], [ %idx.next.i11, %for.body.i12 ]
; CHECK-NEXT:  %dst.i.i7 = getelementptr inbounds double, double* %1, i64 %idx.i6
; CHECK-NEXT:  %dst.i.l.i8 = load double, double* %dst.i.i7, align 1
; CHECK-NEXT:  store double 0.000000e+00, double* %dst.i.i7, align 1
; CHECK-NEXT:  %src.i.i9 = getelementptr inbounds double, double* %5, i64 %idx.i6
; CHECK-NEXT:  %src.i.l.i10 = load double, double* %src.i.i9, align 1
; CHECK-NEXT:  %14 = fadd fast double %src.i.l.i10, %dst.i.l.i8
; CHECK-NEXT:  store double %14, double* %src.i.i9, align 1
; CHECK-NEXT:  %idx.next.i11 = add nuw i64 %idx.i6, 1
; CHECK-NEXT:  %15 = icmp eq i64 %12, %idx.next.i11
; CHECK-NEXT:  br i1 %15, label %__enzyme_memcpyadd_doubleda1sa1.exit13, label %for.body.i12

; CHECK: __enzyme_memcpyadd_doubleda1sa1.exit13:           ; preds = %__enzyme_memcpyadd_doubleda1sa1.exit, %for.body.i12
; CHECK-NEXT:   %16 = lshr i64 %num, 3
; CHECK-NEXT:   %17 = {{(icmp eq i64 %16, 0|icmp ult i64 %num, 8)}}
; CHECK-NEXT:   br i1 %17, label %__enzyme_memcpyadd_doubleda1sa1.exit21, label %for.body.i20

; CHECK: for.body.i20:                                     ; preds = %for.body.i20, %__enzyme_memcpyadd_doubleda1sa1.exit13
; CHECK-NEXT:   %idx.i14 = phi i64 [ 0, %__enzyme_memcpyadd_doubleda1sa1.exit13 ], [ %idx.next.i19, %for.body.i20 ]
; CHECK-NEXT:   %dst.i.i15 = getelementptr inbounds double, double* %2, i64 %idx.i14
; CHECK-NEXT:   %dst.i.l.i16 = load double, double* %dst.i.i15, align 1
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i15, align 1
; CHECK-NEXT:   %src.i.i17 = getelementptr inbounds double, double* %6, i64 %idx.i14
; CHECK-NEXT:   %src.i.l.i18 = load double, double* %src.i.i17, align 1
; CHECK-NEXT:   %18 = fadd fast double %src.i.l.i18, %dst.i.l.i16
; CHECK-NEXT:   store double %18, double* %src.i.i17, align 1
; CHECK-NEXT:   %idx.next.i19 = add nuw i64 %idx.i14, 1
; CHECK-NEXT:   %19 = icmp eq i64 %16, %idx.next.i19
; CHECK-NEXT:   br i1 %19, label %__enzyme_memcpyadd_doubleda1sa1.exit21, label %for.body.i20

; CHECK: __enzyme_memcpyadd_doubleda1sa1.exit21:           ; preds = %__enzyme_memcpyadd_doubleda1sa1.exit13, %for.body.i20
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
