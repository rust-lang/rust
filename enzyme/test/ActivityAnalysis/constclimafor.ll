; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=julia_volumerhs__5948 -activity-analysis-inactive-args -o /dev/null | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-ni:10:11:12:13"
target triple = "nvptx64-nvidia-cuda"

@0 = internal addrspace(3) global [25 x float] zeroinitializer, align 32
@1 = internal addrspace(3) global [125 x float] zeroinitializer, align 32
@2 = internal addrspace(3) global [125 x float] zeroinitializer, align 32
@3 = internal addrspace(3) global [125 x float] zeroinitializer, align 32

define void @julia_volumerhs__5948({ [5 x i64], i8 addrspace(1)* } addrspace(11)* nocapture nonnull readonly align 8 dereferenceable(48) %arg, { [5 x i64], i8 addrspace(1)* } addrspace(11)* nocapture nonnull readonly align 8 dereferenceable(48) %arg1, { [5 x i64], i8 addrspace(1)* } addrspace(11)* nocapture nonnull readonly align 8 dereferenceable(48) %arg2, float %arg3, { [2 x i64], i8 addrspace(1)* } addrspace(11)* nocapture nonnull readonly align 8 dereferenceable(24) %arg4, i64 signext %arg5) {
bb:
  %i = alloca [20 x i8], align 16
  %i6 = getelementptr inbounds [20 x i8], [20 x i8]* %i, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 20, i8* nonnull %i6)
  %i7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y(), !range !2
  %i8 = add nuw nsw i32 %i7, 1
  %i9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %i10 = add nuw nsw i32 %i9, 1
  %i11 = bitcast [20 x i8]* %i to float*
  br label %bb12

bb12:                                             ; preds = %bb12, %bb
  %i13 = phi i64 [ 1, %bb ], [ %i17, %bb12 ]
  %i14 = add nsw i64 %i13, -1
  %i15 = getelementptr inbounds float, float* %i11, i64 %i14
  store float 0.000000e+00, float* %i15, align 4, !tbaa !3
  %i16 = icmp eq i64 %i13, 5
  %i17 = add nuw nsw i64 %i13, 1
  br i1 %i16, label %bb18, label %bb12

bb18:                                             ; preds = %bb12
  %i19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !7
  %i20 = zext i32 %i8 to i64
  %i21 = zext i32 %i10 to i64
  %i22 = getelementptr inbounds { [2 x i64], i8 addrspace(1)* }, { [2 x i64], i8 addrspace(1)* } addrspace(11)* %arg4, i64 0, i32 0, i64 0
  %i23 = load i64, i64 addrspace(11)* %i22, align 8, !tbaa !8
  %i24 = icmp sgt i64 %i23, 0
  %i25 = select i1 %i24, i64 %i23, i64 0
  %i26 = add nsw i64 %i20, -1
  %i27 = mul i64 %i25, %i26
  %i28 = getelementptr inbounds { [2 x i64], i8 addrspace(1)* }, { [2 x i64], i8 addrspace(1)* } addrspace(11)* %arg4, i64 0, i32 1
  %i29 = add nsw i64 %i21, -1
  %i30 = add i64 %i29, %i27
  %i31 = bitcast i8 addrspace(1)* addrspace(11)* %i28 to float addrspace(1)* addrspace(11)*
  %i32 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i31, align 8, !tbaa !8
  %i33 = getelementptr inbounds float, float addrspace(1)* %i32, i64 %i30
  %i34 = bitcast float addrspace(1)* %i33 to i32 addrspace(1)*
  %i35 = load i32, i32 addrspace(1)* %i34, align 4, !tbaa !10
  %i36 = mul nuw nsw i64 %i20, 5
  %i37 = add nsw i64 %i36, -6
  %i38 = add nsw i64 %i37, %i21
  %i39 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i38
  %i40 = bitcast float addrspace(3)* %i39 to i32 addrspace(3)*
  store i32 %i35, i32 addrspace(3)* %i40, align 4, !tbaa !13
  %i41 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 0
  %i42 = load i64, i64 addrspace(11)* %i41, align 8
  %i43 = icmp sgt i64 %i42, 0
  %i44 = select i1 %i43, i64 %i42, i64 0
  %i45 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 1
  %i46 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 2
  %i47 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 3
  %i48 = load i64, i64 addrspace(11)* %i45, align 8
  %i49 = icmp sgt i64 %i48, 0
  %i50 = select i1 %i49, i64 %i48, i64 0
  %i51 = load i64, i64 addrspace(11)* %i46, align 8
  %i52 = icmp sgt i64 %i51, 0
  %i53 = select i1 %i52, i64 %i51, i64 0
  %i54 = load i64, i64 addrspace(11)* %i47, align 8
  %i55 = icmp sgt i64 %i54, 0
  %i56 = select i1 %i55, i64 %i54, i64 0
  %i57 = mul i64 %i50, %i44
  %i58 = mul i64 %i44, %i26
  %i59 = zext i32 %i19 to i64
  %i60 = mul i64 %i56, %i59
  %i61 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 1
  %i62 = add i64 %i60, 9
  %i63 = mul i64 %i62, %i53
  %i64 = add i64 %i29, %i58
  %i65 = bitcast i8 addrspace(1)* addrspace(11)* %i61 to float addrspace(1)* addrspace(11)*
  %i66 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i65, align 8
  %i67 = mul i64 %i53, %i59
  %i68 = mul i64 %i67, %i56
  %i69 = add i64 %i60, 3
  %i70 = mul i64 %i69, %i53
  %i71 = add i64 %i60, 6
  %i72 = mul i64 %i71, %i53
  %i73 = add i64 %i60, 2
  %i74 = mul i64 %i73, %i53
  %i75 = add i64 %i60, 5
  %i76 = mul i64 %i75, %i53
  %i77 = add i64 %i60, 8
  %i78 = mul i64 %i77, %i53
  %i79 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 0
  %i80 = load i64, i64 addrspace(11)* %i79, align 8
  %i81 = icmp sgt i64 %i80, 0
  %i82 = select i1 %i81, i64 %i80, i64 0
  %i83 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 1
  %i84 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 2
  %i85 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 3
  %i86 = load i64, i64 addrspace(11)* %i83, align 8
  %i87 = icmp sgt i64 %i86, 0
  %i88 = select i1 %i87, i64 %i86, i64 0
  %i89 = load i64, i64 addrspace(11)* %i84, align 8
  %i90 = icmp sgt i64 %i89, 0
  %i91 = select i1 %i90, i64 %i89, i64 0
  %i92 = load i64, i64 addrspace(11)* %i85, align 8
  %i93 = icmp sgt i64 %i92, 0
  %i94 = select i1 %i93, i64 %i92, i64 0
  %i95 = mul i64 %i88, %i82
  %i96 = mul i64 %i82, %i26
  %i97 = mul i64 %i95, %i91
  %i98 = mul i64 %i94, %i59
  %i99 = mul i64 %i98, %i97
  %i100 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 1
  %i101 = add i64 %i29, %i96
  %i102 = bitcast i8 addrspace(1)* addrspace(11)* %i100 to float addrspace(1)* addrspace(11)*
  %i103 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i102, align 8
  %i104 = add i64 %i98, 2
  %i105 = mul i64 %i104, %i91
  %i106 = add i64 %i98, 3
  %i107 = mul i64 %i106, %i91
  %i108 = getelementptr inbounds [125 x float], [125 x float] addrspace(3)* @1, i64 0, i64 %i38
  %i109 = mul nuw nsw i64 %i21, 5
  %i110 = add nsw i64 %i109, -6
  br label %bb111

bb111:                                            ; preds = %bb209, %bb18
  %i112 = phi i64 [ 1, %bb18 ], [ %i211, %bb209 ]
  call void @llvm.nvvm.barrier0()
  %i113 = add nsw i64 %i112, -1
  %i114 = add i64 %i113, %i63
  %i115 = mul i64 %i57, %i114
  %i116 = add i64 %i64, %i115
  %i117 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i116
  %i118 = load float, float addrspace(1)* %i117, align 4, !tbaa !10
  %i119 = add i64 %i113, %i68
  %i120 = mul i64 %i119, %i50
  %i121 = add i64 %i26, %i120
  %i122 = mul i64 %i121, %i44
  %i123 = add i64 %i29, %i122
  %i124 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i123
  %i125 = load float, float addrspace(1)* %i124, align 4, !tbaa !10
  %i126 = add i64 %i113, %i70
  %i127 = mul i64 %i57, %i126
  %i128 = add i64 %i64, %i127
  %i129 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i128
  %i130 = load float, float addrspace(1)* %i129, align 4, !tbaa !10
  %i131 = add i64 %i113, %i72
  %i132 = mul i64 %i57, %i131
  %i133 = add i64 %i64, %i132
  %i134 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i133
  %i135 = load float, float addrspace(1)* %i134, align 4, !tbaa !10
  %i136 = add i64 %i113, %i74
  %i137 = mul i64 %i57, %i136
  %i138 = add i64 %i64, %i137
  %i139 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i138
  %i140 = load float, float addrspace(1)* %i139, align 4, !tbaa !10
  %i141 = add i64 %i113, %i76
  %i142 = mul i64 %i57, %i141
  %i143 = add i64 %i64, %i142
  %i144 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i143
  %i145 = load float, float addrspace(1)* %i144, align 4, !tbaa !10
  %i146 = add i64 %i113, %i78
  %i147 = mul i64 %i57, %i146
  %i148 = add i64 %i64, %i147
  %i149 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i148
  %i150 = load float, float addrspace(1)* %i149, align 4, !tbaa !10
  %i151 = mul i64 %i95, %i113
  %i152 = add i64 %i101, %i151
  %i153 = add i64 %i152, %i97
  %i154 = add i64 %i153, %i99
  %i155 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i154
  %i156 = load float, float addrspace(1)* %i155, align 4, !tbaa !10
  %i157 = add i64 %i113, %i105
  %i158 = mul i64 %i95, %i157
  %i159 = add i64 %i101, %i158
  %i160 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i159
  %i161 = load float, float addrspace(1)* %i160, align 4, !tbaa !10
  %i162 = add i64 %i113, %i107
  %i163 = mul i64 %i95, %i162
  %i164 = add i64 %i101, %i163
  %i165 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i164
  %i166 = load float, float addrspace(1)* %i165, align 4, !tbaa !10
  %i167 = fmul float %i125, %i156
  %i168 = fmul float %i130, %i161
  %i169 = fmul float %i135, %i166
  %i170 = fadd float %i167, %i168
  %i171 = fadd float %i170, %i169
  %i172 = fmul float %i118, %i171
  store float %i172, float addrspace(3)* %i108, align 4, !tbaa !13
  %i173 = fmul float %i140, %i156
  %i174 = fmul float %i145, %i161
  %i175 = fmul float %i150, %i166
  %i176 = fadd float %i173, %i174
  %i177 = fadd float %i176, %i175
  %i178 = fmul float %i118, %i177
  %i179 = add nuw nsw i64 %i112, -6
  br label %bb180

bb180:                                            ; preds = %bb180, %bb111
  %i181 = phi i64 [ 1, %bb111 ], [ %i192, %bb180 ]
  %i182 = mul nuw nsw i64 %i181, 5
  %i183 = add nsw i64 %i179, %i182
  %i184 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i183
  %i185 = load float, float addrspace(3)* %i184, align 4, !tbaa !13
  %i186 = add nsw i64 %i181, -1
  %i187 = getelementptr inbounds float, float* %i11, i64 %i186
  %i188 = load float, float* %i187, align 4, !tbaa !3
  %i189 = fmul float %i178, %i185
  %i190 = fadd float %i188, %i189
  store float %i190, float* %i187, align 4, !tbaa !3
  %i191 = icmp eq i64 %i181, 5
  %i192 = add nuw nsw i64 %i181, 1
  br i1 %i191, label %bb193, label %bb180

bb193:                                            ; preds = %bb180
  call void @llvm.nvvm.barrier0()
  %i194 = getelementptr inbounds float, float* %i11, i64 %i113
  %i195 = load float, float* %i194, align 4, !tbaa !3
  br label %bb196

bb196:                                            ; preds = %bb196, %bb193
  %i197 = phi float [ %i195, %bb193 ], [ %i206, %bb196 ]
  %i198 = phi i64 [ 1, %bb193 ], [ %i208, %bb196 ]
  %i199 = add i64 %i110, %i198
  %i200 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i199
  %i201 = load float, float addrspace(3)* %i200, align 4, !tbaa !13
  %i202 = add i64 %i37, %i198
  %i203 = getelementptr inbounds [125 x float], [125 x float] addrspace(3)* @1, i64 0, i64 %i202
  %i204 = load float, float addrspace(3)* %i203, align 4, !tbaa !13
  %i205 = fmul float %i201, %i204
  %i206 = fadd float %i197, %i205
  %i207 = icmp eq i64 %i198, 5
  %i208 = add nuw nsw i64 %i198, 1
  br i1 %i207, label %bb209, label %bb196, !llvm.loop !15

bb209:                                            ; preds = %bb196
  store float %i206, float* %i194, align 4, !tbaa !3
  %i210 = icmp eq i64 %i112, 5
  %i211 = add nuw nsw i64 %i112, 1
  br i1 %i210, label %bb212, label %bb111, !llvm.loop !17

bb212:                                            ; preds = %bb209
  %i213 = add i64 %i60, 10
  %i214 = mul i64 %i213, %i53
  %i215 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 0
  %i216 = load i64, i64 addrspace(11)* %i215, align 8, !tbaa !8
  %i217 = icmp sgt i64 %i216, 0
  %i218 = select i1 %i217, i64 %i216, i64 0
  %i219 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 1
  %i220 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 2
  %i221 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 3
  %i222 = load i64, i64 addrspace(11)* %i219, align 8, !tbaa !8
  %i223 = icmp sgt i64 %i222, 0
  %i224 = select i1 %i223, i64 %i222, i64 0
  %i225 = load i64, i64 addrspace(11)* %i220, align 8, !tbaa !8
  %i226 = icmp sgt i64 %i225, 0
  %i227 = select i1 %i226, i64 %i225, i64 0
  %i228 = load i64, i64 addrspace(11)* %i221, align 8, !tbaa !8
  %i229 = icmp sgt i64 %i228, 0
  %i230 = select i1 %i229, i64 %i228, i64 0
  %i231 = mul i64 %i227, %i59
  %i232 = mul i64 %i231, %i230
  %i233 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 1
  %i234 = bitcast i8 addrspace(1)* addrspace(11)* %i233 to float addrspace(1)* addrspace(11)*
  %i235 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i234, align 8, !tbaa !8
  br label %bb236

bb236:                                            ; preds = %bb236, %bb212
  %i237 = phi i64 [ %i256, %bb236 ], [ 1, %bb212 ]
  %i238 = add nsw i64 %i237, -1
  %i239 = add i64 %i238, %i214
  %i240 = mul i64 %i57, %i239
  %i241 = add i64 %i64, %i240
  %i242 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i241
  %i243 = load float, float addrspace(1)* %i242, align 4, !tbaa !10
  %i244 = add i64 %i238, %i232
  %i245 = mul i64 %i244, %i224
  %i246 = add i64 %i26, %i245
  %i247 = mul i64 %i246, %i218
  %i248 = add i64 %i29, %i247
  %i249 = getelementptr inbounds float, float addrspace(1)* %i235, i64 %i248
  %i250 = load float, float addrspace(1)* %i249, align 4, !tbaa !10
  %i251 = getelementptr inbounds float, float* %i11, i64 %i238
  %i252 = load float, float* %i251, align 4, !tbaa !3
  %i253 = fmul float %i243, %i252
  %i254 = fadd float %i250, %i253
  store float %i254, float addrspace(1)* %i249, align 4, !tbaa !10
  %i255 = icmp eq i64 %i237, 5
  %i256 = add nuw nsw i64 %i237, 1
  br i1 %i255, label %bb257, label %bb236, !llvm.loop !18

bb257:                                            ; preds = %bb236
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z() #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{i32 0, i32 1023}
!3 = !{!4, !4, i64 0}
!4 = !{!"jtbaa_data", !5, i64 0}
!5 = !{!"jtbaa", !6, i64 0}
!6 = !{!"jtbaa"}
!7 = !{i32 0, i32 2147483646}
!8 = !{!9, !9, i64 0}
!9 = !{!"jtbaa_const", !5, i64 0}
!10 = !{!11, !11, i64 0, i64 0}
!11 = !{!"custom_tbaa_addrspace(1)", !12, i64 0}
!12 = !{!"custom_tbaa"}
!13 = !{!14, !14, i64 0, i64 0}
!14 = !{!"custom_tbaa_addrspace(3)", !12, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.unroll.full"}
!17 = distinct !{!17, !16}
!18 = distinct !{!18, !16}

; CHECK: { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg: icv:1
; CHECK-NEXT: { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1: icv:1
; CHECK-NEXT: { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2: icv:1
; CHECK-NEXT: float %arg3: icv:1
; CHECK-NEXT: { [2 x i64], i8 addrspace(1)* } addrspace(11)* %arg4: icv:1
; CHECK-NEXT: i64 %arg5: icv:1
; CHECK-NEXT: bb
; CHECK-NEXT:   %i = alloca [20 x i8], align 16: icv:1 ici:1
; CHECK-NEXT:   %i6 = getelementptr inbounds [20 x i8], [20 x i8]* %i, i64 0, i64 0: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 20, i8* nonnull %i6): icv:1 ici:1
; CHECK-NEXT:   %i7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y(), !range !2: icv:1 ici:1
; CHECK-NEXT:   %i8 = add nuw nsw i32 %i7, 1: icv:1 ici:1
; CHECK-NEXT:   %i9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2: icv:1 ici:1
; CHECK-NEXT:   %i10 = add nuw nsw i32 %i9, 1: icv:1 ici:1
; CHECK-NEXT:   %i11 = bitcast [20 x i8]* %i to float*: icv:1 ici:1
; CHECK-NEXT:   br label %bb12: icv:1 ici:1
; CHECK-NEXT: bb12
; CHECK-NEXT:   %i13 = phi i64 [ 1, %bb ], [ %i17, %bb12 ]: icv:1 ici:1
; CHECK-NEXT:   %i14 = add nsw i64 %i13, -1: icv:1 ici:1
; CHECK-NEXT:   %i15 = getelementptr inbounds float, float* %i11, i64 %i14: icv:1 ici:1
; CHECK-NEXT:   store float 0.000000e+00, float* %i15, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   %i16 = icmp eq i64 %i13, 5: icv:1 ici:1
; CHECK-NEXT:   %i17 = add nuw nsw i64 %i13, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i16, label %bb18, label %bb12: icv:1 ici:1
; CHECK-NEXT: bb18
; CHECK-NEXT:   %i19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !7: icv:1 ici:1
; CHECK-NEXT:   %i20 = zext i32 %i8 to i64: icv:1 ici:1
; CHECK-NEXT:   %i21 = zext i32 %i10 to i64: icv:1 ici:1
; CHECK-NEXT:   %i22 = getelementptr inbounds { [2 x i64], i8 addrspace(1)* }, { [2 x i64], i8 addrspace(1)* } addrspace(11)* %arg4, i64 0, i32 0, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i23 = load i64, i64 addrspace(11)* %i22, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i24 = icmp sgt i64 %i23, 0: icv:1 ici:1
; CHECK-NEXT:   %i25 = select i1 %i24, i64 %i23, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i26 = add nsw i64 %i20, -1: icv:1 ici:1
; CHECK-NEXT:   %i27 = mul i64 %i25, %i26: icv:1 ici:1
; CHECK-NEXT:   %i28 = getelementptr inbounds { [2 x i64], i8 addrspace(1)* }, { [2 x i64], i8 addrspace(1)* } addrspace(11)* %arg4, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   %i29 = add nsw i64 %i21, -1: icv:1 ici:1
; CHECK-NEXT:   %i30 = add i64 %i29, %i27: icv:1 ici:1
; CHECK-NEXT:   %i31 = bitcast i8 addrspace(1)* addrspace(11)* %i28 to float addrspace(1)* addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %i32 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i31, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i33 = getelementptr inbounds float, float addrspace(1)* %i32, i64 %i30: icv:1 ici:1
; CHECK-NEXT:   %i34 = bitcast float addrspace(1)* %i33 to i32 addrspace(1)*: icv:1 ici:1
; CHECK-NEXT:   %i35 = load i32, i32 addrspace(1)* %i34, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i36 = mul nuw nsw i64 %i20, 5: icv:1 ici:1
; CHECK-NEXT:   %i37 = add nsw i64 %i36, -6: icv:1 ici:1
; CHECK-NEXT:   %i38 = add nsw i64 %i37, %i21: icv:1 ici:1
; CHECK-NEXT:   %i39 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i38: icv:1 ici:1
; CHECK-NEXT:   %i40 = bitcast float addrspace(3)* %i39 to i32 addrspace(3)*: icv:1 ici:1
; CHECK-NEXT:   store i32 %i35, i32 addrspace(3)* %i40, align 4, !tbaa !13: icv:1 ici:1
; CHECK-NEXT:   %i41 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i42 = load i64, i64 addrspace(11)* %i41, align 8: icv:1 ici:1
; CHECK-NEXT:   %i43 = icmp sgt i64 %i42, 0: icv:1 ici:1
; CHECK-NEXT:   %i44 = select i1 %i43, i64 %i42, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i45 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 1: icv:1 ici:1
; CHECK-NEXT:   %i46 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 2: icv:1 ici:1
; CHECK-NEXT:   %i47 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 0, i64 3: icv:1 ici:1
; CHECK-NEXT:   %i48 = load i64, i64 addrspace(11)* %i45, align 8: icv:1 ici:1
; CHECK-NEXT:   %i49 = icmp sgt i64 %i48, 0: icv:1 ici:1
; CHECK-NEXT:   %i50 = select i1 %i49, i64 %i48, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i51 = load i64, i64 addrspace(11)* %i46, align 8: icv:1 ici:1
; CHECK-NEXT:   %i52 = icmp sgt i64 %i51, 0: icv:1 ici:1
; CHECK-NEXT:   %i53 = select i1 %i52, i64 %i51, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i54 = load i64, i64 addrspace(11)* %i47, align 8: icv:1 ici:1
; CHECK-NEXT:   %i55 = icmp sgt i64 %i54, 0: icv:1 ici:1
; CHECK-NEXT:   %i56 = select i1 %i55, i64 %i54, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i57 = mul i64 %i50, %i44: icv:1 ici:1
; CHECK-NEXT:   %i58 = mul i64 %i44, %i26: icv:1 ici:1
; CHECK-NEXT:   %i59 = zext i32 %i19 to i64: icv:1 ici:1
; CHECK-NEXT:   %i60 = mul i64 %i56, %i59: icv:1 ici:1
; CHECK-NEXT:   %i61 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg2, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   %i62 = add i64 %i60, 9: icv:1 ici:1
; CHECK-NEXT:   %i63 = mul i64 %i62, %i53: icv:1 ici:1
; CHECK-NEXT:   %i64 = add i64 %i29, %i58: icv:1 ici:1
; CHECK-NEXT:   %i65 = bitcast i8 addrspace(1)* addrspace(11)* %i61 to float addrspace(1)* addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %i66 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i65, align 8: icv:1 ici:1
; CHECK-NEXT:   %i67 = mul i64 %i53, %i59: icv:1 ici:1
; CHECK-NEXT:   %i68 = mul i64 %i67, %i56: icv:1 ici:1
; CHECK-NEXT:   %i69 = add i64 %i60, 3: icv:1 ici:1
; CHECK-NEXT:   %i70 = mul i64 %i69, %i53: icv:1 ici:1
; CHECK-NEXT:   %i71 = add i64 %i60, 6: icv:1 ici:1
; CHECK-NEXT:   %i72 = mul i64 %i71, %i53: icv:1 ici:1
; CHECK-NEXT:   %i73 = add i64 %i60, 2: icv:1 ici:1
; CHECK-NEXT:   %i74 = mul i64 %i73, %i53: icv:1 ici:1
; CHECK-NEXT:   %i75 = add i64 %i60, 5: icv:1 ici:1
; CHECK-NEXT:   %i76 = mul i64 %i75, %i53: icv:1 ici:1
; CHECK-NEXT:   %i77 = add i64 %i60, 8: icv:1 ici:1
; CHECK-NEXT:   %i78 = mul i64 %i77, %i53: icv:1 ici:1
; CHECK-NEXT:   %i79 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i80 = load i64, i64 addrspace(11)* %i79, align 8: icv:1 ici:1
; CHECK-NEXT:   %i81 = icmp sgt i64 %i80, 0: icv:1 ici:1
; CHECK-NEXT:   %i82 = select i1 %i81, i64 %i80, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i83 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 1: icv:1 ici:1
; CHECK-NEXT:   %i84 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 2: icv:1 ici:1
; CHECK-NEXT:   %i85 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 0, i64 3: icv:1 ici:1
; CHECK-NEXT:   %i86 = load i64, i64 addrspace(11)* %i83, align 8: icv:1 ici:1
; CHECK-NEXT:   %i87 = icmp sgt i64 %i86, 0: icv:1 ici:1
; CHECK-NEXT:   %i88 = select i1 %i87, i64 %i86, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i89 = load i64, i64 addrspace(11)* %i84, align 8: icv:1 ici:1
; CHECK-NEXT:   %i90 = icmp sgt i64 %i89, 0: icv:1 ici:1
; CHECK-NEXT:   %i91 = select i1 %i90, i64 %i89, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i92 = load i64, i64 addrspace(11)* %i85, align 8: icv:1 ici:1
; CHECK-NEXT:   %i93 = icmp sgt i64 %i92, 0: icv:1 ici:1
; CHECK-NEXT:   %i94 = select i1 %i93, i64 %i92, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i95 = mul i64 %i88, %i82: icv:1 ici:1
; CHECK-NEXT:   %i96 = mul i64 %i82, %i26: icv:1 ici:1
; CHECK-NEXT:   %i97 = mul i64 %i95, %i91: icv:1 ici:1
; CHECK-NEXT:   %i98 = mul i64 %i94, %i59: icv:1 ici:1
; CHECK-NEXT:   %i99 = mul i64 %i98, %i97: icv:1 ici:1
; CHECK-NEXT:   %i100 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg1, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   %i101 = add i64 %i29, %i96: icv:1 ici:1
; CHECK-NEXT:   %i102 = bitcast i8 addrspace(1)* addrspace(11)* %i100 to float addrspace(1)* addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %i103 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i102, align 8: icv:1 ici:1
; CHECK-NEXT:   %i104 = add i64 %i98, 2: icv:1 ici:1
; CHECK-NEXT:   %i105 = mul i64 %i104, %i91: icv:1 ici:1
; CHECK-NEXT:   %i106 = add i64 %i98, 3: icv:1 ici:1
; CHECK-NEXT:   %i107 = mul i64 %i106, %i91: icv:1 ici:1
; CHECK-NEXT:   %i108 = getelementptr inbounds [125 x float], [125 x float] addrspace(3)* @1, i64 0, i64 %i38: icv:1 ici:1
; CHECK-NEXT:   %i109 = mul nuw nsw i64 %i21, 5: icv:1 ici:1
; CHECK-NEXT:   %i110 = add nsw i64 %i109, -6: icv:1 ici:1
; CHECK-NEXT:   br label %bb111: icv:1 ici:1
; CHECK-NEXT: bb111
; CHECK-NEXT:   %i112 = phi i64 [ 1, %bb18 ], [ %i211, %bb209 ]: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.nvvm.barrier0(): icv:1 ici:1
; CHECK-NEXT:   %i113 = add nsw i64 %i112, -1: icv:1 ici:1
; CHECK-NEXT:   %i114 = add i64 %i113, %i63: icv:1 ici:1
; CHECK-NEXT:   %i115 = mul i64 %i57, %i114: icv:1 ici:1
; CHECK-NEXT:   %i116 = add i64 %i64, %i115: icv:1 ici:1
; CHECK-NEXT:   %i117 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i116: icv:1 ici:1
; CHECK-NEXT:   %i118 = load float, float addrspace(1)* %i117, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i119 = add i64 %i113, %i68: icv:1 ici:1
; CHECK-NEXT:   %i120 = mul i64 %i119, %i50: icv:1 ici:1
; CHECK-NEXT:   %i121 = add i64 %i26, %i120: icv:1 ici:1
; CHECK-NEXT:   %i122 = mul i64 %i121, %i44: icv:1 ici:1
; CHECK-NEXT:   %i123 = add i64 %i29, %i122: icv:1 ici:1
; CHECK-NEXT:   %i124 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i123: icv:1 ici:1
; CHECK-NEXT:   %i125 = load float, float addrspace(1)* %i124, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i126 = add i64 %i113, %i70: icv:1 ici:1
; CHECK-NEXT:   %i127 = mul i64 %i57, %i126: icv:1 ici:1
; CHECK-NEXT:   %i128 = add i64 %i64, %i127: icv:1 ici:1
; CHECK-NEXT:   %i129 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i128: icv:1 ici:1
; CHECK-NEXT:   %i130 = load float, float addrspace(1)* %i129, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i131 = add i64 %i113, %i72: icv:1 ici:1
; CHECK-NEXT:   %i132 = mul i64 %i57, %i131: icv:1 ici:1
; CHECK-NEXT:   %i133 = add i64 %i64, %i132: icv:1 ici:1
; CHECK-NEXT:   %i134 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i133: icv:1 ici:1
; CHECK-NEXT:   %i135 = load float, float addrspace(1)* %i134, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i136 = add i64 %i113, %i74: icv:1 ici:1
; CHECK-NEXT:   %i137 = mul i64 %i57, %i136: icv:1 ici:1
; CHECK-NEXT:   %i138 = add i64 %i64, %i137: icv:1 ici:1
; CHECK-NEXT:   %i139 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i138: icv:1 ici:1
; CHECK-NEXT:   %i140 = load float, float addrspace(1)* %i139, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i141 = add i64 %i113, %i76: icv:1 ici:1
; CHECK-NEXT:   %i142 = mul i64 %i57, %i141: icv:1 ici:1
; CHECK-NEXT:   %i143 = add i64 %i64, %i142: icv:1 ici:1
; CHECK-NEXT:   %i144 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i143: icv:1 ici:1
; CHECK-NEXT:   %i145 = load float, float addrspace(1)* %i144, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i146 = add i64 %i113, %i78: icv:1 ici:1
; CHECK-NEXT:   %i147 = mul i64 %i57, %i146: icv:1 ici:1
; CHECK-NEXT:   %i148 = add i64 %i64, %i147: icv:1 ici:1
; CHECK-NEXT:   %i149 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i148: icv:1 ici:1
; CHECK-NEXT:   %i150 = load float, float addrspace(1)* %i149, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i151 = mul i64 %i95, %i113: icv:1 ici:1
; CHECK-NEXT:   %i152 = add i64 %i101, %i151: icv:1 ici:1
; CHECK-NEXT:   %i153 = add i64 %i152, %i97: icv:1 ici:1
; CHECK-NEXT:   %i154 = add i64 %i153, %i99: icv:1 ici:1
; CHECK-NEXT:   %i155 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i154: icv:1 ici:1
; CHECK-NEXT:   %i156 = load float, float addrspace(1)* %i155, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i157 = add i64 %i113, %i105: icv:1 ici:1
; CHECK-NEXT:   %i158 = mul i64 %i95, %i157: icv:1 ici:1
; CHECK-NEXT:   %i159 = add i64 %i101, %i158: icv:1 ici:1
; CHECK-NEXT:   %i160 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i159: icv:1 ici:1
; CHECK-NEXT:   %i161 = load float, float addrspace(1)* %i160, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i162 = add i64 %i113, %i107: icv:1 ici:1
; CHECK-NEXT:   %i163 = mul i64 %i95, %i162: icv:1 ici:1
; CHECK-NEXT:   %i164 = add i64 %i101, %i163: icv:1 ici:1
; CHECK-NEXT:   %i165 = getelementptr inbounds float, float addrspace(1)* %i103, i64 %i164: icv:1 ici:1
; CHECK-NEXT:   %i166 = load float, float addrspace(1)* %i165, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i167 = fmul float %i125, %i156: icv:1 ici:1
; CHECK-NEXT:   %i168 = fmul float %i130, %i161: icv:1 ici:1
; CHECK-NEXT:   %i169 = fmul float %i135, %i166: icv:1 ici:1
; CHECK-NEXT:   %i170 = fadd float %i167, %i168: icv:1 ici:1
; CHECK-NEXT:   %i171 = fadd float %i170, %i169: icv:1 ici:1
; CHECK-NEXT:   %i172 = fmul float %i118, %i171: icv:1 ici:1
; CHECK-NEXT:   store float %i172, float addrspace(3)* %i108, align 4, !tbaa !13: icv:1 ici:1
; CHECK-NEXT:   %i173 = fmul float %i140, %i156: icv:1 ici:1
; CHECK-NEXT:   %i174 = fmul float %i145, %i161: icv:1 ici:1
; CHECK-NEXT:   %i175 = fmul float %i150, %i166: icv:1 ici:1
; CHECK-NEXT:   %i176 = fadd float %i173, %i174: icv:1 ici:1
; CHECK-NEXT:   %i177 = fadd float %i176, %i175: icv:1 ici:1
; CHECK-NEXT:   %i178 = fmul float %i118, %i177: icv:1 ici:1
; CHECK-NEXT:   %i179 = add nuw nsw i64 %i112, -6: icv:1 ici:1
; CHECK-NEXT:   br label %bb180: icv:1 ici:1
; CHECK-NEXT: bb180
; CHECK-NEXT:   %i181 = phi i64 [ 1, %bb111 ], [ %i192, %bb180 ]: icv:1 ici:1
; CHECK-NEXT:   %i182 = mul nuw nsw i64 %i181, 5: icv:1 ici:1
; CHECK-NEXT:   %i183 = add nsw i64 %i179, %i182: icv:1 ici:1
; CHECK-NEXT:   %i184 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i183: icv:1 ici:1
; CHECK-NEXT:   %i185 = load float, float addrspace(3)* %i184, align 4, !tbaa !13: icv:1 ici:1
; CHECK-NEXT:   %i186 = add nsw i64 %i181, -1: icv:1 ici:1
; CHECK-NEXT:   %i187 = getelementptr inbounds float, float* %i11, i64 %i186: icv:1 ici:1
; CHECK-NEXT:   %i188 = load float, float* %i187, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   %i189 = fmul float %i178, %i185: icv:1 ici:1
; CHECK-NEXT:   %i190 = fadd float %i188, %i189: icv:1 ici:1
; CHECK-NEXT:   store float %i190, float* %i187, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   %i191 = icmp eq i64 %i181, 5: icv:1 ici:1
; CHECK-NEXT:   %i192 = add nuw nsw i64 %i181, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i191, label %bb193, label %bb180: icv:1 ici:1
; CHECK-NEXT: bb193
; CHECK-NEXT:   call void @llvm.nvvm.barrier0(): icv:1 ici:1
; CHECK-NEXT:   %i194 = getelementptr inbounds float, float* %i11, i64 %i113: icv:1 ici:1
; CHECK-NEXT:   %i195 = load float, float* %i194, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   br label %bb196: icv:1 ici:1
; CHECK-NEXT: bb196
; CHECK-NEXT:   %i197 = phi float [ %i195, %bb193 ], [ %i206, %bb196 ]: icv:1 ici:1
; CHECK-NEXT:   %i198 = phi i64 [ 1, %bb193 ], [ %i208, %bb196 ]: icv:1 ici:1
; CHECK-NEXT:   %i199 = add i64 %i110, %i198: icv:1 ici:1
; CHECK-NEXT:   %i200 = getelementptr inbounds [25 x float], [25 x float] addrspace(3)* @0, i64 0, i64 %i199: icv:1 ici:1
; CHECK-NEXT:   %i201 = load float, float addrspace(3)* %i200, align 4, !tbaa !13: icv:1 ici:1
; CHECK-NEXT:   %i202 = add i64 %i37, %i198: icv:1 ici:1
; CHECK-NEXT:   %i203 = getelementptr inbounds [125 x float], [125 x float] addrspace(3)* @1, i64 0, i64 %i202: icv:1 ici:1
; CHECK-NEXT:   %i204 = load float, float addrspace(3)* %i203, align 4, !tbaa !13: icv:1 ici:1
; CHECK-NEXT:   %i205 = fmul float %i201, %i204: icv:1 ici:1
; CHECK-NEXT:   %i206 = fadd float %i197, %i205: icv:1 ici:1
; CHECK-NEXT:   %i207 = icmp eq i64 %i198, 5: icv:1 ici:1
; CHECK-NEXT:   %i208 = add nuw nsw i64 %i198, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i207, label %bb209, label %bb196, !llvm.loop !15: icv:1 ici:1
; CHECK-NEXT: bb209
; CHECK-NEXT:   store float %i206, float* %i194, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   %i210 = icmp eq i64 %i112, 5: icv:1 ici:1
; CHECK-NEXT:   %i211 = add nuw nsw i64 %i112, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i210, label %bb212, label %bb111, !llvm.loop !17: icv:1 ici:1
; CHECK-NEXT: bb212
; CHECK-NEXT:   %i213 = add i64 %i60, 10: icv:1 ici:1
; CHECK-NEXT:   %i214 = mul i64 %i213, %i53: icv:1 ici:1
; CHECK-NEXT:   %i215 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i216 = load i64, i64 addrspace(11)* %i215, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i217 = icmp sgt i64 %i216, 0: icv:1 ici:1
; CHECK-NEXT:   %i218 = select i1 %i217, i64 %i216, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i219 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 1: icv:1 ici:1
; CHECK-NEXT:   %i220 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 2: icv:1 ici:1
; CHECK-NEXT:   %i221 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 0, i64 3: icv:1 ici:1
; CHECK-NEXT:   %i222 = load i64, i64 addrspace(11)* %i219, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i223 = icmp sgt i64 %i222, 0: icv:1 ici:1
; CHECK-NEXT:   %i224 = select i1 %i223, i64 %i222, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i225 = load i64, i64 addrspace(11)* %i220, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i226 = icmp sgt i64 %i225, 0: icv:1 ici:1
; CHECK-NEXT:   %i227 = select i1 %i226, i64 %i225, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i228 = load i64, i64 addrspace(11)* %i221, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   %i229 = icmp sgt i64 %i228, 0: icv:1 ici:1
; CHECK-NEXT:   %i230 = select i1 %i229, i64 %i228, i64 0: icv:1 ici:1
; CHECK-NEXT:   %i231 = mul i64 %i227, %i59: icv:1 ici:1
; CHECK-NEXT:   %i232 = mul i64 %i231, %i230: icv:1 ici:1
; CHECK-NEXT:   %i233 = getelementptr inbounds { [5 x i64], i8 addrspace(1)* }, { [5 x i64], i8 addrspace(1)* } addrspace(11)* %arg, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   %i234 = bitcast i8 addrspace(1)* addrspace(11)* %i233 to float addrspace(1)* addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %i235 = load float addrspace(1)*, float addrspace(1)* addrspace(11)* %i234, align 8, !tbaa !8: icv:1 ici:1
; CHECK-NEXT:   br label %bb236: icv:1 ici:1
; CHECK-NEXT: bb236
; CHECK-NEXT:   %i237 = phi i64 [ %i256, %bb236 ], [ 1, %bb212 ]: icv:1 ici:1
; CHECK-NEXT:   %i238 = add nsw i64 %i237, -1: icv:1 ici:1
; CHECK-NEXT:   %i239 = add i64 %i238, %i214: icv:1 ici:1
; CHECK-NEXT:   %i240 = mul i64 %i57, %i239: icv:1 ici:1
; CHECK-NEXT:   %i241 = add i64 %i64, %i240: icv:1 ici:1
; CHECK-NEXT:   %i242 = getelementptr inbounds float, float addrspace(1)* %i66, i64 %i241: icv:1 ici:1
; CHECK-NEXT:   %i243 = load float, float addrspace(1)* %i242, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i244 = add i64 %i238, %i232: icv:1 ici:1
; CHECK-NEXT:   %i245 = mul i64 %i244, %i224: icv:1 ici:1
; CHECK-NEXT:   %i246 = add i64 %i26, %i245: icv:1 ici:1
; CHECK-NEXT:   %i247 = mul i64 %i246, %i218: icv:1 ici:1
; CHECK-NEXT:   %i248 = add i64 %i29, %i247: icv:1 ici:1
; CHECK-NEXT:   %i249 = getelementptr inbounds float, float addrspace(1)* %i235, i64 %i248: icv:1 ici:1
; CHECK-NEXT:   %i250 = load float, float addrspace(1)* %i249, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i251 = getelementptr inbounds float, float* %i11, i64 %i238: icv:1 ici:1
; CHECK-NEXT:   %i252 = load float, float* %i251, align 4, !tbaa !3: icv:1 ici:1
; CHECK-NEXT:   %i253 = fmul float %i243, %i252: icv:1 ici:1
; CHECK-NEXT:   %i254 = fadd float %i250, %i253: icv:1 ici:1
; CHECK-NEXT:   store float %i254, float addrspace(1)* %i249, align 4, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %i255 = icmp eq i64 %i237, 5: icv:1 ici:1
; CHECK-NEXT:   %i256 = add nuw nsw i64 %i237, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i255, label %bb257, label %bb236, !llvm.loop !18: icv:1 ici:1
; CHECK-NEXT: bb257
; CHECK-NEXT:   ret void: icv:1 ici:1