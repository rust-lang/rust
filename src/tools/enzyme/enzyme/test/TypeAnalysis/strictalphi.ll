; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=f -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s

declare i8* @_Znwm(i64)

define void @f() {
e:
  %i78 = call noalias nonnull i8* @_Znwm(i64 8)
  br label %bb155

bb155:                                            
  %i159 = phi i8* [ %i78, %e ], [ %i220, %bb216 ]
  %l = load i8,  i8* %i159, align 1
  br i1 true, label %bb179, label %bb216

bb179: 
  %i192 = call noalias nonnull i8* @_Znwm(i64 8)
  br label %bb216

bb216: 
  %i217 = phi i8* [ %i192, %bb179 ], [ %i159, %bb155 ]
  %i220 = getelementptr inbounds i8, i8* %i217, i64 1
  br i1 true, label %bb153, label %bb155

bb153:                                            ; preds = %bb216
  ret void
}

; CHECK: f - {} |
; CHECK-NEXT: e
; CHECK-NEXT:   %i78 = call noalias nonnull i8* @_Znwm(i64 8): {[-1]:Pointer}
; CHECK-NEXT:   br label %bb155: {}
; CHECK-NEXT: bb155
; CHECK-NEXT:   %i159 = phi i8* [ %i78, %e ], [ %i220, %bb216 ]: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   %l = load i8, i8* %i159, align 1: {[-1]:Integer}
; CHECK-NEXT:   br i1 true, label %bb179, label %bb216: {}
; CHECK-NEXT: bb179
; CHECK-NEXT:   %i192 = call noalias nonnull i8* @_Znwm(i64 8): {[-1]:Pointer}
; CHECK-NEXT:   br label %bb216: {}
; CHECK-NEXT: bb216
; CHECK-NEXT:   %i217 = phi i8* [ %i192, %bb179 ], [ %i159, %bb155 ]: {[-1]:Pointer}
; CHECK-NEXT:   %i220 = getelementptr inbounds i8, i8* %i217, i64 1: {[-1]:Pointer}
; CHECK-NEXT:   br i1 true, label %bb153, label %bb155: {}
; CHECK-NEXT: bb153
; CHECK-NEXT:   ret void: {}
