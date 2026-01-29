//@ [COPY] compile-flags: --cfg=copy
//@ revisions: COPY CLONE

// Test case from https://github.com/rust-lang/rust/issues/128081.
// Ensure both Copy and Clone get optimized copy.

#[unsafe(no_mangle)]
pub fn intra_clone(intra: &Av1BlockIntra) -> Av1BlockIntraInter {
    // CHECK-LABEL: fn intra_clone(
    // CHECK: [[C:_.*]] = copy (*_1);
    // CHECK: _0 = Av1BlockIntraInter::Intra(move [[C]]);
    Av1BlockIntraInter::Intra(intra.clone())
}

#[unsafe(no_mangle)]
pub fn inter_clone(inter: &Av1BlockInter) -> Av1BlockIntraInter {
    // CHECK-LABEL: fn inter_clone(
    // CHECK: [[C:_.*]] = copy (*_1);
    // CHECK: _0 = Av1BlockIntraInter::Inter(move [[C]]);
    Av1BlockIntraInter::Inter(inter.clone())
}

#[unsafe(no_mangle)]
pub fn dav1dsequenceheader_copy(v: &Dav1dSequenceHeader) -> Dav1dSequenceHeader {
    // CHECK-LABEL: fn dav1dsequenceheader_copy(
    // CHECK: _0 = copy (*_1);
    v.clone()
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct mv {
    pub y: i16,
    pub x: i16,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct MaskedInterIntraPredMode(u8);

#[derive(Clone)]
#[cfg_attr(copy, derive(Copy))]
#[repr(C)]
pub struct Av1BlockInter1d {
    pub mv: [mv; 2],
    pub wedge_idx: u8,
    pub mask_sign: u8,
    pub interintra_mode: MaskedInterIntraPredMode,
    pub _padding: u8,
}

#[derive(Clone)]
#[cfg_attr(copy, derive(Copy))]
#[repr(C)]
pub struct Av1BlockInterNd {
    pub one_d: Av1BlockInter1d,
}

#[derive(Clone, Copy)]
pub enum CompInterType {
    WeightedAvg = 1,
    Avg = 2,
    Seg = 3,
    Wedge = 4,
}

#[derive(Clone, Copy)]
pub enum MotionMode {
    Translation = 0,
    Obmc = 1,
    Warp = 2,
}

#[derive(Clone, Copy)]
pub enum DrlProximity {
    Nearest,
    Nearer,
    Near,
    Nearish,
}

#[derive(Clone, Copy)]
pub enum TxfmSize {
    S4x4 = 0,
    S8x8 = 1,
    S16x16 = 2,
    S32x32 = 3,
    S64x64 = 4,
    R4x8 = 5,
    R8x4 = 6,
    R8x16 = 7,
    R16x8 = 8,
    R16x32 = 9,
    R32x16 = 10,
    R32x64 = 11,
    R64x32 = 12,
    R4x16 = 13,
    R16x4 = 14,
    R8x32 = 15,
    R32x8 = 16,
    R16x64 = 17,
    R64x16 = 18,
}

#[derive(Clone, Copy)]
pub enum Filter2d {
    Regular8Tap = 0,
    RegularSmooth8Tap = 1,
    RegularSharp8Tap = 2,
    SharpRegular8Tap = 3,
    SharpSmooth8Tap = 4,
    Sharp8Tap = 5,
    SmoothRegular8Tap = 6,
    Smooth8Tap = 7,
    SmoothSharp8Tap = 8,
    Bilinear = 9,
}

#[derive(Clone, Copy)]
pub enum InterIntraType {
    Blend,
    Wedge,
}

#[cfg_attr(copy, derive(Copy))]
#[derive(Clone)]
#[repr(C)]
pub struct Av1BlockInter {
    pub nd: Av1BlockInterNd,
    pub comp_type: Option<CompInterType>,
    pub inter_mode: u8,
    pub motion_mode: MotionMode,
    pub drl_idx: DrlProximity,
    pub r#ref: [i8; 2],
    pub max_ytx: TxfmSize,
    pub filter2d: Filter2d,
    pub interintra_type: Option<InterIntraType>,
    pub tx_split0: u8,
    pub tx_split1: u16,
}

#[cfg_attr(copy, derive(Copy))]
#[derive(Clone)]
#[repr(C)]
pub struct Av1BlockIntra {
    pub y_mode: u8,
    pub uv_mode: u8,
    pub tx: TxfmSize,
    pub pal_sz: [u8; 2],
    pub y_angle: i8,
    pub uv_angle: i8,
    pub cfl_alpha: [i8; 2],
}

#[repr(C)]
pub enum Av1BlockIntraInter {
    Intra(Av1BlockIntra),
    Inter(Av1BlockInter),
}

use std::ffi::{c_int, c_uint};

pub type Dav1dPixelLayout = c_uint;
pub type Dav1dColorPrimaries = c_uint;
pub type Dav1dTransferCharacteristics = c_uint;
pub type Dav1dMatrixCoefficients = c_uint;
pub type Dav1dChromaSamplePosition = c_uint;
pub type Dav1dAdaptiveBoolean = c_uint;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Dav1dSequenceHeaderOperatingPoint {
    pub major_level: u8,
    pub minor_level: u8,
    pub initial_display_delay: u8,
    pub idc: u16,
    pub tier: u8,
    pub decoder_model_param_present: u8,
    pub display_model_param_present: u8,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Dav1dSequenceHeaderOperatingParameterInfo {
    pub decoder_buffer_delay: u32,
    pub encoder_buffer_delay: u32,
    pub low_delay_mode: u8,
}

pub const DAV1D_MAX_OPERATING_POINTS: usize = 32;

#[cfg_attr(copy, derive(Copy))]
#[derive(Clone)]
#[repr(C)]
pub struct Dav1dSequenceHeader {
    pub profile: u8,
    pub max_width: c_int,
    pub max_height: c_int,
    pub layout: Dav1dPixelLayout,
    pub pri: Dav1dColorPrimaries,
    pub trc: Dav1dTransferCharacteristics,
    pub mtrx: Dav1dMatrixCoefficients,
    pub chr: Dav1dChromaSamplePosition,
    pub hbd: u8,
    pub color_range: u8,
    pub num_operating_points: u8,
    pub operating_points: [Dav1dSequenceHeaderOperatingPoint; DAV1D_MAX_OPERATING_POINTS],
    pub still_picture: u8,
    pub reduced_still_picture_header: u8,
    pub timing_info_present: u8,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: u8,
    pub num_ticks_per_picture: u32,
    pub decoder_model_info_present: u8,
    pub encoder_decoder_buffer_delay_length: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_delay_length: u8,
    pub frame_presentation_delay_length: u8,
    pub display_model_info_present: u8,
    pub width_n_bits: u8,
    pub height_n_bits: u8,
    pub frame_id_numbers_present: u8,
    pub delta_frame_id_n_bits: u8,
    pub frame_id_n_bits: u8,
    pub sb128: u8,
    pub filter_intra: u8,
    pub intra_edge_filter: u8,
    pub inter_intra: u8,
    pub masked_compound: u8,
    pub warped_motion: u8,
    pub dual_filter: u8,
    pub order_hint: u8,
    pub jnt_comp: u8,
    pub ref_frame_mvs: u8,
    pub screen_content_tools: Dav1dAdaptiveBoolean,
    pub force_integer_mv: Dav1dAdaptiveBoolean,
    pub order_hint_n_bits: u8,
    pub super_res: u8,
    pub cdef: u8,
    pub restoration: u8,
    pub ss_hor: u8,
    pub ss_ver: u8,
    pub monochrome: u8,
    pub color_description_present: u8,
    pub separate_uv_delta_q: u8,
    pub film_grain_present: u8,
    pub operating_parameter_info:
        [Dav1dSequenceHeaderOperatingParameterInfo; DAV1D_MAX_OPERATING_POINTS],
}
