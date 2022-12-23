; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define internal fastcc void @f({} addrspace(10)* %arg) {
bb:
  br label %bb54

bb54:                                             ; preds = %bb91, %bb42
  %i55 = phi i64 [ 0, %bb ], [ %i69, %bb91 ]
  %i60 = icmp slt i64 %i55, 4
  br i1 %i60, label %bb66, label %bb92

bb66:                                             ; preds = %bb86, %bb61
  %i67 = phi i64 [ %i55, %bb54 ], [ %i69, %bb84 ]
  %i68 = phi {} addrspace(10)* [ null, %bb54 ], [ %i90, %bb84 ]
  %i69 = add nsw i64 %i67, 1
  %i71 = icmp eq {} addrspace(10)* %i68, %arg
  br i1 %i71, label %bb72, label %bb91

bb72:                                             ; preds = %bb66
  %i74 = call {}* @julia.pointer_from_objref({} addrspace(10)* %i68)
  %i79 = icmp eq {}* %i74, null
  br i1 %i79, label %bb84, label %bb91

bb84:                                             ; preds = %bb72
  %i85 = icmp slt i64 %i67, 3
  %i90 = call {} addrspace(10)* @__dynamic_cast()
  br i1 %i85, label %bb66, label %bb92

bb91:                                             ; preds = %bb66, %bb72
  br label %bb54

bb92:                                             ; preds = %bb84, %bb54, %bb35
  ret void
}

; Function Attrs: nofree nounwind readnone
declare nonnull {}* @julia.pointer_from_objref({} addrspace(10)*) local_unnamed_addr #4

; Function Attrs: nofree readonly
declare nonnull {} addrspace(10)* @__dynamic_cast()


declare dso_local void @__enzyme_autodiff(...)

define void @dsquare() local_unnamed_addr {
bb:
  call void (...) @__enzyme_autodiff(i8* bitcast (void ({} addrspace(10)*)* @f to i8*), metadata !"enzyme_dup", {} addrspace(10)* null, {} addrspace(10)* null)
  ret void
}

attributes #4 = { nounwind readnone }


; CHECK: invertbb91: 
; CHECK-NEXT:   %[[i31:.+]] = load i64, i64* %"iv'ac"
; CHECK-NEXT:   %[[i32:.+]] = load {} addrspace(10)**, {} addrspace(10)*** %"i68!manual_lcssa_cache", align 8
; CHECK-NEXT:   %[[i33:.+]] = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)** %[[i32]], i64 %[[i31]]
; CHECK-NEXT:   %[[i34:.+]] = load {} addrspace(10)*, {} addrspace(10)** %[[i33]], align 8
; CHECK-NEXT:   %i71_unwrap = icmp eq {} addrspace(10)* %[[i34]], %arg
; CHECK-NEXT:   br i1 %i71_unwrap, label %mergeinvertbb66_bb91, label %mergeinvertbb66_bb911


; OLD: invertbb91: 
; OLD-NEXT:   %[[i78:.+]] = load i64, i64* %"iv'ac"
; OLD-NEXT:   %[[i79:.+]] = load i64*, i64** %loopLimit_cache, align 8
; OLD-NEXT:   %[[i80:.+]] = getelementptr inbounds i64, i64* %[[i79]], i64 %[[i78]]
; OLD-NEXT:   %[[i81:.+]] = load i64, i64* %[[i80]], align 8
; OLD-NEXT:   %[[i82:.+]] = icmp ne i64 %[[i81]], 0
; OLD-NEXT:   br i1 %[[i82]], label %invertbb91_phirc, label %invertbb91_phirc6

; OLD: invertbb91_phirc:
; OLD-NEXT:   %[[i83:.+]] = sub nuw i64 %[[i81]], 1
; OLD-NEXT:   %[[i84:.+]] = load {} addrspace(10)***, {} addrspace(10)**** %i90_cache, align 8
; OLD-NEXT:   %[[i85:.+]] = getelementptr inbounds {} addrspace(10)**, {} addrspace(10)*** %[[i84]], i64 %[[i78]]
; OLD-NEXT:   %[[i86:.+]] = load {} addrspace(10)**, {} addrspace(10)*** %[[i85]], align 8
; OLD-NEXT:   %[[i87:.+]] = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)** %[[i86]], i64 %[[i83]]
; OLD-NEXT:   %[[i88:.+]] = load {} addrspace(10)*, {} addrspace(10)** %[[i87]], align 8
; OLD-NEXT:   br label %invertbb91_phimerge

; OLD: invertbb91_phirc6:  
; OLD-NEXT:   br label %invertbb91_phimerge

; OLD: invertbb91_phimerge: 
; OLD-NEXT:   %[[i89:.+]] = phi {} addrspace(10)* [ %[[i88]], %invertbb91_phirc ], [ null, %invertbb91_phirc6 ]
; OLD-NEXT:   %i71_unwrap = icmp eq {} addrspace(10)* %[[i89]], %arg
