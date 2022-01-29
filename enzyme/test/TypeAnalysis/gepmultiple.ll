; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

declare i1 @cmp()

define void @callee(i8* nocapture %data) {
entry:
  %call = tail call i1 @cmp()
  br i1 %call, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i8, i8* %data, i64 1
  store i8 97, i8* %arrayidx, align 1, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %idx.0 = phi i64 [ 1, %if.then ], [ 0, %entry ]
  %call1 = tail call i1 @cmp()
  br i1 %call1, label %if.end7, label %if.then3

if.then3:                                         ; preds = %if.end
  %add4 = add nuw nsw i64 %idx.0, 1
  %arrayidx5 = getelementptr inbounds i8, i8* %data, i64 %add4
  store i8 97, i8* %arrayidx5, align 1, !tbaa !2
  br label %if.end7

if.end7:                                          ; preds = %if.then3, %if.end
  %idx.1 = phi i64 [ %add4, %if.then3 ], [ %idx.0, %if.end ]
  %call2 = tail call i1 @cmp()
  br i1 %call2, label %if.end14, label %if.then10

if.then10:                                        ; preds = %if.end7
  %add11 = add nuw nsw i64 %idx.1, 1
  %arrayidx12 = getelementptr inbounds i8, i8* %data, i64 %add11
  store i8 97, i8* %arrayidx12, align 1, !tbaa !2
  br label %if.end14

if.end14:                                         ; preds = %if.then10, %if.end7
  %idx.2 = phi i64 [ %add11, %if.then10 ], [ %idx.1, %if.end7 ]
  %call3 = tail call i1 @cmp()
  br i1 %call3, label %if.end21, label %if.then17

if.then17:                                        ; preds = %if.end14
  %add18 = add nuw nsw i64 %idx.2, 1
  %arrayidx19 = getelementptr inbounds i8, i8* %data, i64 %add18
  store i8 97, i8* %arrayidx19, align 1, !tbaa !2
  br label %if.end21

if.end21:                                         ; preds = %if.then17, %if.end14
  %idx.3 = phi i64 [ %add18, %if.then17 ], [ %idx.2, %if.end14 ]
  %call4 = tail call i1 @cmp()
  br i1 %call4, label %if.end28, label %if.then24

if.then24:                                        ; preds = %if.end21
  %add25 = add nuw nsw i64 %idx.3, 1
  %arrayidx26 = getelementptr inbounds i8, i8* %data, i64 %add25
  store i8 97, i8* %arrayidx26, align 1, !tbaa !2
  br label %if.end28

if.end28:                                         ; preds = %if.then24, %if.end21
  %idx.4 = phi i64 [ %add25, %if.then24 ], [ %idx.3, %if.end21 ]
  %call5 = tail call i1 @cmp()
  br i1 %call5, label %if.end35, label %if.then31

if.then31:                                        ; preds = %if.end28
  %add32 = add nuw nsw i64 %idx.4, 1
  %arrayidx33 = getelementptr inbounds i8, i8* %data, i64 %add32
  store i8 97, i8* %arrayidx33, align 1, !tbaa !2
  br label %if.end35

if.end35:                                         ; preds = %if.then31, %if.end28
  %idx.5 = phi i64 [ %add32, %if.then31 ], [ %idx.4, %if.end28 ]
  %call6 = tail call i1 @cmp()
  br i1 %call6, label %if.end41, label %if.then38

if.then38:                                        ; preds = %if.end35
  %add39 = add nuw nsw i64 %idx.5, 1
  %arrayidx40 = getelementptr inbounds i8, i8* %data, i64 %add39
  store i8 97, i8* %arrayidx40, align 1, !tbaa !2
  br label %if.end41

if.end41:                                         ; preds = %if.then38, %if.end35
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}

; CHECK: callee - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i8* %data: {[-1]:Pointer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call, label %if.end, label %if.then: {}
; CHECK-NEXT: if.then
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i8, i8* %data, i64 1: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end: {}
; CHECK-NEXT: if.end
; CHECK-NEXT:   %idx.0 = phi i64 [ 1, %if.then ], [ 0, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %call1 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call1, label %if.end7, label %if.then3: {}
; CHECK-NEXT: if.then3
; CHECK-NEXT:   %add4 = add nuw nsw i64 %idx.0, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds i8, i8* %data, i64 %add4: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx5, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end7: {}
; CHECK-NEXT: if.end7
; CHECK-NEXT:   %idx.1 = phi i64 [ %add4, %if.then3 ], [ %idx.0, %if.end ]: {[-1]:Integer}
; CHECK-NEXT:   %call2 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call2, label %if.end14, label %if.then10: {}
; CHECK-NEXT: if.then10
; CHECK-NEXT:   %add11 = add nuw nsw i64 %idx.1, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx12 = getelementptr inbounds i8, i8* %data, i64 %add11: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx12, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end14: {}
; CHECK-NEXT: if.end14
; CHECK-NEXT:   %idx.2 = phi i64 [ %add11, %if.then10 ], [ %idx.1, %if.end7 ]: {[-1]:Integer}
; CHECK-NEXT:   %call3 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call3, label %if.end21, label %if.then17: {}
; CHECK-NEXT: if.then17
; CHECK-NEXT:   %add18 = add nuw nsw i64 %idx.2, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx19 = getelementptr inbounds i8, i8* %data, i64 %add18: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx19, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end21: {}
; CHECK-NEXT: if.end21
; CHECK-NEXT:   %idx.3 = phi i64 [ %add18, %if.then17 ], [ %idx.2, %if.end14 ]: {[-1]:Integer}
; CHECK-NEXT:   %call4 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call4, label %if.end28, label %if.then24: {}
; CHECK-NEXT: if.then24
; CHECK-NEXT:   %add25 = add nuw nsw i64 %idx.3, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx26 = getelementptr inbounds i8, i8* %data, i64 %add25: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx26, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end28: {}
; CHECK-NEXT: if.end28
; CHECK-NEXT:   %idx.4 = phi i64 [ %add25, %if.then24 ], [ %idx.3, %if.end21 ]: {[-1]:Integer}
; CHECK-NEXT:   %call5 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call5, label %if.end35, label %if.then31: {}
; CHECK-NEXT: if.then31
; CHECK-NEXT:   %add32 = add nuw nsw i64 %idx.4, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx33 = getelementptr inbounds i8, i8* %data, i64 %add32: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx33, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end35: {}
; CHECK-NEXT: if.end35
; CHECK-NEXT:   %idx.5 = phi i64 [ %add32, %if.then31 ], [ %idx.4, %if.end28 ]: {[-1]:Integer}
; CHECK-NEXT:   %call6 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call6, label %if.end41, label %if.then38: {}
; CHECK-NEXT: if.then38
; CHECK-NEXT:   %add39 = add nuw nsw i64 %idx.5, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx40 = getelementptr inbounds i8, i8* %data, i64 %add39: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx40, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end41: {}
; CHECK-NEXT: if.end41
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: callee - {} |{[-1]:Pointer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}:{}
; CHECK-NEXT: i8* %data: {[-1]:Pointer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call, label %if.end, label %if.then: {}
; CHECK-NEXT: if.then
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i8, i8* %data, i64 1: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end: {}
; CHECK-NEXT: if.end
; CHECK-NEXT:   %idx.0 = phi i64 [ 1, %if.then ], [ 0, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %call1 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call1, label %if.end7, label %if.then3: {}
; CHECK-NEXT: if.then3
; CHECK-NEXT:   %add4 = add nuw nsw i64 %idx.0, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds i8, i8* %data, i64 %add4: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx5, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end7: {}
; CHECK-NEXT: if.end7
; CHECK-NEXT:   %idx.1 = phi i64 [ %add4, %if.then3 ], [ %idx.0, %if.end ]: {[-1]:Integer}
; CHECK-NEXT:   %call2 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call2, label %if.end14, label %if.then10: {}
; CHECK-NEXT: if.then10
; CHECK-NEXT:   %add11 = add nuw nsw i64 %idx.1, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx12 = getelementptr inbounds i8, i8* %data, i64 %add11: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx12, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end14: {}
; CHECK-NEXT: if.end14
; CHECK-NEXT:   %idx.2 = phi i64 [ %add11, %if.then10 ], [ %idx.1, %if.end7 ]: {[-1]:Integer}
; CHECK-NEXT:   %call3 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call3, label %if.end21, label %if.then17: {}
; CHECK-NEXT: if.then17
; CHECK-NEXT:   %add18 = add nuw nsw i64 %idx.2, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx19 = getelementptr inbounds i8, i8* %data, i64 %add18: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx19, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end21: {}
; CHECK-NEXT: if.end21
; CHECK-NEXT:   %idx.3 = phi i64 [ %add18, %if.then17 ], [ %idx.2, %if.end14 ]: {[-1]:Integer}
; CHECK-NEXT:   %call4 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call4, label %if.end28, label %if.then24: {}
; CHECK-NEXT: if.then24
; CHECK-NEXT:   %add25 = add nuw nsw i64 %idx.3, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx26 = getelementptr inbounds i8, i8* %data, i64 %add25: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx26, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end28: {}
; CHECK-NEXT: if.end28
; CHECK-NEXT:   %idx.4 = phi i64 [ %add25, %if.then24 ], [ %idx.3, %if.end21 ]: {[-1]:Integer}
; CHECK-NEXT:   %call5 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call5, label %if.end35, label %if.then31: {}
; CHECK-NEXT: if.then31
; CHECK-NEXT:   %add32 = add nuw nsw i64 %idx.4, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx33 = getelementptr inbounds i8, i8* %data, i64 %add32: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx33, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end35: {}
; CHECK-NEXT: if.end35
; CHECK-NEXT:   %idx.5 = phi i64 [ %add32, %if.then31 ], [ %idx.4, %if.end28 ]: {[-1]:Integer}
; CHECK-NEXT:   %call6 = tail call i1 @cmp(): {[-1]:Integer}
; CHECK-NEXT:   br i1 %call6, label %if.end41, label %if.then38: {}
; CHECK-NEXT: if.then38
; CHECK-NEXT:   %add39 = add nuw nsw i64 %idx.5, 1: {[-1]:Integer}
; CHECK-NEXT:   %arrayidx40 = getelementptr inbounds i8, i8* %data, i64 %add39: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   store i8 97, i8* %arrayidx40, align 1, !tbaa !2: {}
; CHECK-NEXT:   br label %if.end41: {}
; CHECK-NEXT: if.end41
; CHECK-NEXT:   ret void: {}
