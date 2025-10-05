use crate::parser::item::OUTPUT_NAME;

// Proper primitive types
// i8, i16, i32, i64, i128 and isize
// u8, u16, u32, u64, u128 and usize
// f32, f64
// char
// bool
// () -- why is this possible for parameters :/

pub static I8: &str = "i8";
pub static I16: &str = "i16";
pub static I32: &str = "i32";
pub static I64: &str = "i64";
pub static I128: &str = "i128";
pub static ISIZE: &str = "isize";

pub static U8: &str = "u8";
pub static U16: &str = "u16";
pub static U32: &str = "u32";
pub static U64: &str = "u64";
pub static U128: &str = "u128";
pub static USIZE: &str = "usize";

pub static F32: &str = "f32";
pub static F64: &str = "f64";

pub static CHAR: &str = "char";
pub static BOOL: &str = "bool";
pub static UNIT: &str = "()";
pub static STR: &str = "str";
pub static STRING: &str = "String";
pub static VEC: &str = "Vec";

// placeholders are between the strs

pub(crate) static INIT_NONCE: &str = "fn __skip() { let mut __daikon_nonce = 0;\nlet mut __unwrap_nonce = NONCE_COUNTER.lock().unwrap();\n__daikon_nonce = *__unwrap_nonce;\n*__unwrap_nonce += 1;\ndrop(__unwrap_nonce);\n }";
pub(crate) fn init_nonce() -> String {
    String::from(INIT_NONCE)
}

// TODO: before build_entry, initialize __daikon_nonce with the NONCE_COUNTER lock.
pub(crate) static DTRACE_ENTRY: [&str; 2] =
    ["fn __skip() { dtrace_entry(\"", ":::ENTER\", __daikon_nonce); }"];
pub(crate) fn build_entry(ppt_name: String) -> String {
    let mut res = String::from(DTRACE_ENTRY[0]);
    res.push_str(&ppt_name);
    res.push_str(DTRACE_ENTRY[1]);
    res
}

pub(crate) static DTRACE_EXIT: [&str; 3] =
    ["fn __skip() { dtrace_exit(\"", ":::EXIT", "\", __daikon_nonce); }"];
pub(crate) fn build_exit(ppt_name: String, exit_counter: usize) -> String {
    let mut res = String::from(DTRACE_EXIT[0]);
    res.push_str(&ppt_name);
    res.push_str(DTRACE_EXIT[1]);
    res.push_str(&exit_counter.to_string());
    res.push_str(DTRACE_EXIT[2]);
    res
}

pub(crate) static DTRACE_PRIM: [&str; 4] =
    ["fn __skip() { dtrace_print_prim::<", ">(", ", String::from(\"", "\")); }"];
pub(crate) fn build_prim(p_type: String, var_name: String) -> String {
    let mut res = String::from(DTRACE_PRIM[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM[3]);
    res
}

pub(crate) static DTRACE_PRIM_RET: [&str; 2] =
    ["fn __skip() { dtrace_print_prim::<", ">(__daikon_ret, String::from(\"return\")); }"];
pub(crate) fn build_prim_ret(p_type: String) -> String {
    let mut res = String::from(DTRACE_PRIM_RET[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_RET[1]);
    res
}

pub(crate) static DTRACE_PRIM_REF: [&str; 5] = [
    "fn __skip() { dtrace_print_prim::<",
    ">(",
    "::from_str(&",
    ".to_string()).expect(\"Ok\"), String::from(\"",
    "\")); }",
];
pub(crate) fn build_prim_ref(p_type: String, var_name: String) -> String {
    let mut res = String::from(DTRACE_PRIM_REF[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF[1]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM_REF[3]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM_REF[4]);
    res
}

// Generic routine used to print all non-string primitive parameter and return
// values.
pub(crate) static DTRACE_PRIM_REF_RET: [&str; 3] = [
    "fn __skip() { dtrace_print_prim::<",
    ">(",
    "::from_str(&__daikon_ret.to_string()).expect(\"Ok\"), String::from(\"return\")); }",
];
pub(crate) fn build_prim_ref_ret(p_type: String) -> String {
    let mut res = String::from(DTRACE_PRIM_REF_RET[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF_RET[1]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF_RET[2]);
    res
}

pub(crate) static DTRACE_PRIM_TOSTRING: [&str; 3] =
    ["fn __skip() { dtrace_print_string(", ".to_string(), String::from(\"", "\")); }"];
pub(crate) fn build_prim_with_tostring(var_name: String) -> String {
    // TODO: change name
    let mut res = String::from(DTRACE_PRIM_TOSTRING[0]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM_TOSTRING[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRIM_TOSTRING[2]);
    res
}

pub(crate) static DTRACE_PRIM_TOSTRING_RET: &str =
    "fn __skip() { dtrace_print_string(__daikon_ret.to_string(), String::from(\"return\")); }";
pub(crate) fn build_prim_with_tostring_ret() -> String {
    String::from(DTRACE_PRIM_TOSTRING_RET)
}

pub(crate) static DTRACE_PRIM_FIELD_TOSTRING: [&str; 3] =
    ["dtrace_print_string(self.", ".to_string(), format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_prim_field_tostring(field_name: String) -> String {
    let mut res = String::from(DTRACE_PRIM_FIELD_TOSTRING[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_FIELD_TOSTRING[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_FIELD_TOSTRING[2]);
    res
}

// pub(crate) fn build_prim_with_to_string

pub(crate) static DTRACE_PRIM_STRUCT: [&str; 4] =
    ["dtrace_print_prim::<", ">(self.", ", format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_field_prim(p_type: String, field_name: String) -> String {
    let mut res = String::from(DTRACE_PRIM_STRUCT[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_STRUCT[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_STRUCT[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_STRUCT[3]);
    res
}

// TODO: if you have Vec<&'a &'b i32>, you will probably have to make a new Vec<i32> like this
//       to satisfy dtrace_print_prim_vec<T>(v: &Vec<T>).
pub(crate) static DTRACE_PRIM_REF_STRUCT: [&str; 5] = [
    "dtrace_print_prim::<",
    ">(",
    "::from_str(&self.",
    ".to_string()).expect(\"Ok\"), format!(\"{}{}\", prefix, \".",
    "\"));",
];
pub(crate) fn build_field_prim_ref(p_type: String, field_name: String) -> String {
    let mut res = String::from(DTRACE_PRIM_REF_STRUCT[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF_STRUCT[1]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRIM_REF_STRUCT[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_REF_STRUCT[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRIM_REF_STRUCT[4]);
    res
}

pub(crate) static DTRACE_USERDEF: [&str; 6] = [
    "fn __skip() { dtrace_print_pointer(",
    " as *const _ as usize, String::from(\"",
    "\"));\n",
    ".dtrace_print_fields(",
    ", String::from(\"",
    "\")); }",
];
pub(crate) fn build_userdef(var_name: String, depth_arg: i32) -> String {
    let mut res = String::from(DTRACE_USERDEF[0]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[3]);
    res.push_str(&String::from(depth_arg.to_string()));
    res.push_str(DTRACE_USERDEF[4]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[5]);
    res
}

pub(crate) fn build_userdef_with_ampersand_access(var_name: String, depth_arg: i32) -> String {
    let mut res = String::from(DTRACE_USERDEF[0]);
    res.push_str(&format!("&{}", var_name));
    res.push_str(DTRACE_USERDEF[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[3]);
    res.push_str(&String::from(depth_arg.to_string()));
    res.push_str(DTRACE_USERDEF[4]);
    res.push_str(&var_name);
    res.push_str(DTRACE_USERDEF[5]);
    res
}

pub(crate) static DTRACE_USERDEF_RET: [&str; 2] = [
    "fn __skip() { dtrace_print_pointer(__daikon_ret as *const _ as usize, String::from(\"return\"));\n__daikon_ret.dtrace_print_fields(",
    ", String::from(\"return\")); }",
];
pub(crate) fn build_userdef_ret(depth_arg: i32) -> String {
    let mut res = String::from(DTRACE_USERDEF_RET[0]);
    res.push_str(&String::from(depth_arg.to_string()));
    res.push_str(DTRACE_USERDEF_RET[1]);
    res
}

pub(crate) static DTRACE_USERDEF_RET_AMPERSAND: [&str; 2] = [
    "fn __skip() { dtrace_print_pointer(&__daikon_ret as *const _ as usize, String::from(\"return\"));\n__daikon_ret.dtrace_print_fields(",
    ", String::from(\"return\")); }",
];
pub(crate) fn build_userdef_ret_ampersand(depth_arg: i32) -> String {
    let mut res = String::from(DTRACE_USERDEF_RET_AMPERSAND[0]);
    res.push_str(&String::from(depth_arg.to_string()));
    res.push_str(DTRACE_USERDEF_RET_AMPERSAND[1]);
    res
}

pub(crate) static DTRACE_USERDEF_STRUCT: [&str; 5] = [
    "dtrace_print_pointer(self.",
    " as *const _ as usize, format!(\"{}{}\", prefix, \".",
    "\"));\nself.",
    ".dtrace_print_fields(depth - 1, format!(\"{}{}\", prefix, \".",
    "\"));",
];
pub(crate) fn build_field_userdef(field_name: String) -> String {
    let mut res = String::from(DTRACE_USERDEF_STRUCT[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT[4]);
    res
}

pub(crate) static DTRACE_USERDEF_STRUCT_AMPERSAND: [&str; 5] = [
    "dtrace_print_pointer(&self.",
    " as *const _ as usize, format!(\"{}{}\", prefix, \".",
    "\"));\nself.",
    ".dtrace_print_fields(depth - 1, format!(\"{}{}\", prefix, \".",
    "\"));",
];
pub(crate) fn build_field_userdef_with_ampersand_access(field_name: String) -> String {
    let mut res = String::from(DTRACE_USERDEF_STRUCT_AMPERSAND[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT_AMPERSAND[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT_AMPERSAND[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT_AMPERSAND[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_STRUCT_AMPERSAND[4]);
    res
}

// always with ampersand, we will always make a copy.
pub(crate) static DTRACE_USERDEF_VEC_FIELDS: [&str; 3] =
    ["::dtrace_print_fields_vec(&", ", depth - 1, format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_print_vec_fields_userdef(
    plain_struct: String,
    tmp_vec_name: String,
    field_name: String,
) -> String {
    let mut res = String::from(&plain_struct);
    res.push_str(DTRACE_USERDEF_VEC_FIELDS[0]);
    res.push_str(&tmp_vec_name);
    res.push_str(DTRACE_USERDEF_VEC_FIELDS[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_USERDEF_VEC_FIELDS[2]);
    res
}

pub(crate) static DTRACE_PRINT_XFIELD_VEC: [&str; 4] =
    ["::dtrace_print_", "_vec(&", ", format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_print_xfield_vec(
    plain_struct: String,
    field_name: String,
    tmp_vec_name: String,
) -> String {
    let mut res = String::from(&plain_struct);
    res.push_str(DTRACE_PRINT_XFIELD_VEC[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELD_VEC[1]);
    res.push_str(&tmp_vec_name);
    res.push_str(DTRACE_PRINT_XFIELD_VEC[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELD_VEC[3]);
    res
}

pub(crate) static DTRACE_PRINT_POINTER_VEC_USERDEF: [&str; 4] =
    ["dtrace_print_pointer_vec::<", ">(&", ", format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_print_pointer_vec_userdef(
    plain_struct: String,
    tmp_vec_name: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_PRINT_POINTER_VEC_USERDEF[0]);
    res.push_str(&plain_struct);
    res.push_str(DTRACE_PRINT_POINTER_VEC_USERDEF[1]);
    res.push_str(&tmp_vec_name);
    res.push_str(DTRACE_PRINT_POINTER_VEC_USERDEF[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_POINTER_VEC_USERDEF[3]);
    res
}

// we're expecting a tmp vec loop before this.
pub(crate) static DTRACE_VEC_POINTER: [&str; 3] =
    ["dtrace_print_pointer(", ".as_ptr() as usize, String::from(\"", "\"));"];
pub(crate) fn build_pointer_vec(var_name: String) -> String {
    let mut res = String::from(DTRACE_VEC_POINTER[0]);
    res.push_str(&var_name);
    res.push_str(DTRACE_VEC_POINTER[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_VEC_POINTER[2]);
    res
}

pub(crate) static DTRACE_VEC_POINTER_RET: &str =
    "dtrace_print_pointer(__daikon_ret.as_ptr() as usize, String::from(\"return\"));";
pub(crate) fn build_pointer_vec_ret() -> String {
    String::from(DTRACE_VEC_POINTER_RET)
}

pub(crate) static DTRACE_PRINT_POINTER_VEC: [&str; 4] = [
    "dtrace_print_pointer_vec::<",
    ">(&",
    ", format!(\"{}{}\", String::from(\"",
    "\"), \"[..]\"));",
];
pub(crate) fn build_print_pointer_vec(
    basic_type: String,
    tmp_name: String,
    var_name: String,
) -> String {
    let mut res = String::from(DTRACE_PRINT_POINTER_VEC[0]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_PRINT_POINTER_VEC[1]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_POINTER_VEC[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRINT_POINTER_VEC[3]);
    res
}

pub(crate) static DTRACE_VEC_FIELDS: [&str; 3] =
    ["::dtrace_print_fields_vec(&", ", 3, format!(\"{}{}\", String::from(\"", "\"), \"[..]\")); }"]; // we always mash with a tmp loop, so just close __skip.
pub(crate) fn build_print_vec_fields(
    plain_struct: String,
    tmp_vec_name: String,
    var_name: String,
) -> String {
    let mut res = plain_struct.clone();
    res.push_str(DTRACE_VEC_FIELDS[0]);
    res.push_str(&tmp_vec_name);
    res.push_str(DTRACE_VEC_FIELDS[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_VEC_FIELDS[2]);
    res
}

pub(crate) static DAIKON_TMP_VEC_USERDEF: [&str; 8] = [
    "let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < v.len() { __daikon_tmp",
    ".push(v[__daikon_tmp",
    "]); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_daikon_tmp_vec_userdef(
    first_tmp: String,
    basic_type: String,
    next_tmp: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC_USERDEF[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[1]);
    res.push_str(&basic_type);
    res.push_str(DAIKON_TMP_VEC_USERDEF[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[4]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[5]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[6]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF[7]);
    res
}

pub(crate) static DAIKON_TMP_VEC_USERDEF_FIELD: [&str; 9] = [
    "let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < v.len() { __daikon_tmp",
    ".push(v[__daikon_tmp",
    "].",
    "); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_daikon_tmp_vec_field_userdef(
    first_tmp: String,
    field_type: String,
    next_tmp: String,
    field_name: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC_USERDEF_FIELD[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[1]);
    res.push_str(&field_type);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[4]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[5]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[6]);
    res.push_str(&field_name);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[7]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD[8]);
    res
}

pub(crate) static DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND: [&str; 9] = [
    "let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < v.len() { __daikon_tmp",
    ".push(&v[__daikon_tmp",
    "].",
    "); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_daikon_tmp_vec_field_userdef_ampersand(
    first_tmp: String,
    field_type: String,
    next_tmp: String,
    field_name: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[1]);
    res.push_str(&field_type);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[4]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[5]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[6]);
    res.push_str(&field_name);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[7]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_USERDEF_FIELD_AMPERSAND[8]);
    res
}

// this will always be mashed with some subsequent call, so don't close __skip yet.
// the thing you mash it with must close __skip.
pub(crate) static DAIKON_TMP_VEC: [&str; 10] = [
    "fn __skip() { let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < ",
    ".len() { __daikon_tmp",
    ".push(",
    "[__daikon_tmp",
    "]); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_daikon_tmp_vec(
    first_tmp: String,
    basic_type: String,
    next_tmp: String,
    var_name: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC[1]);
    res.push_str(&basic_type);
    res.push_str(DAIKON_TMP_VEC[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC[4]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC[5]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC[6]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC[7]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC[8]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC[9]);
    res
}

// TODO: use this for params/returns where you have Vec<Type>.
pub(crate) static DAIKON_TMP_VEC_AMPERSAND: [&str; 10] = [
    "fn __skip() { let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < ",
    ".len() { __daikon_tmp",
    ".push(&", // for Vec<Type>, need to take &.
    "[__daikon_tmp",
    "]); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_daikon_tmp_vec_ampersand(
    first_tmp: String,
    basic_type: String,
    next_tmp: String,
    var_name: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC_AMPERSAND[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[1]);
    res.push_str(&basic_type);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[4]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[5]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[6]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[7]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[8]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_AMPERSAND[9]);
    res
}

pub(crate) static DAIKON_TMP_VEC_PRIM: [&str; 11] = [
    "fn __skip() { let mut __daikon_tmp",
    ": Vec<",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < ",
    ".len() {__daikon_tmp",
    ".push(",
    "::from_str(&",
    "[__daikon_tmp",
    "].to_string()).expect(\"Ok\")); __daikon_tmp",
    " += 1; }",
]; // don't close __skip() because we will mash
pub(crate) fn build_tmp_vec_prim(
    first_tmp: String,
    p_type: String,
    next_tmp: String,
    var_name: String,
) -> String {
    let mut res = String::from(DAIKON_TMP_VEC_PRIM[0]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[1]);
    res.push_str(&p_type);
    res.push_str(DAIKON_TMP_VEC_PRIM[2]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[3]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[4]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC_PRIM[5]);
    res.push_str(&first_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[6]);
    res.push_str(&p_type);
    res.push_str(DAIKON_TMP_VEC_PRIM[7]);
    res.push_str(&var_name);
    res.push_str(DAIKON_TMP_VEC_PRIM[8]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[9]);
    res.push_str(&next_tmp);
    res.push_str(DAIKON_TMP_VEC_PRIM[10]);
    res
}

pub(crate) static DTRACE_TMP_VEC_FOR_FIELD: [&str; 10] = [
    "let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < self.",
    ".len() { __daikon_tmp",
    ".push(self.",
    "[__daikon_tmp",
    "]); __daikon_tmp",
    " += 1 }",
];
pub(crate) fn build_tmp_vec_for_field(
    first_tmp: String,
    basic_type: String,
    next_tmp: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_TMP_VEC_FOR_FIELD[0]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[1]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[2]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[3]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[5]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[6]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[7]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[8]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD[9]);
    res
}

// TODO: use this for fields which are f: Vec<Type> or f: &Vec<Type>, need to use &.
pub(crate) static DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND: [&str; 10] = [
    "let mut __daikon_tmp",
    ": Vec<&",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < self.",
    ".len() { __daikon_tmp",
    ".push(&self.", // for f: Vec<Type>.
    "[__daikon_tmp",
    "]); __daikon_tmp",
    " += 1 }",
];
pub(crate) fn build_tmp_vec_for_field_ampersand(
    first_tmp: String,
    basic_type: String,
    next_tmp: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[0]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[1]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[2]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[3]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[5]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[6]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[7]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[8]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_VEC_FOR_FIELD_AMPERSAND[9]);
    res
}

pub(crate) static DTRACE_POINTER_VEC_USERDEF: [&str; 3] =
    ["dtrace_print_pointer(self.", ".as_ptr() as usize, format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_pointer_vec_userdef(field_name: String) -> String {
    let mut res = String::from(DTRACE_POINTER_VEC_USERDEF[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_POINTER_VEC_USERDEF[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_POINTER_VEC_USERDEF[2]);
    res
}

pub(crate) static DTRACE_POINTER_ARR_USERDEF: [&str; 3] = [
    "dtrace_print_pointer(self.",
    " as *const _ as *const () as usize, format!(\"{}{}\", prefix, \".",
    "\"));",
];
pub(crate) fn build_pointer_arr_userdef(field_name: String) -> String {
    let mut res = String::from(DTRACE_POINTER_ARR_USERDEF[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_POINTER_ARR_USERDEF[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_POINTER_ARR_USERDEF[2]);
    res
}

pub(crate) static DTRACE_POINTERS_VEC_USERDEF: [&str; 4] =
    ["dtrace_print_pointer_vec::<", ">(&", ", format!(\"{}{}[..]\", prefix, \".", "\"));"];
pub(crate) fn build_pointers_vec_userdef(
    basic_type: String,
    tmp_name: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_POINTERS_VEC_USERDEF[0]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_POINTERS_VEC_USERDEF[1]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_POINTERS_VEC_USERDEF[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_POINTERS_VEC_USERDEF[3]);
    res
}

pub(crate) static DTRACE_PRINT_FIELDS_FOR_FIELD: [&str; 3] =
    ["::dtrace_print_fields_vec(&", ", depth - 1, format!(\"{}{}[..]\", prefix, \".", "\"));"];
pub(crate) fn build_print_vec_fields_for_field(
    basic_type: String,
    tmp_name: String,
    field_name: String,
) -> String {
    let mut res = basic_type.clone();
    res.push_str(DTRACE_PRINT_FIELDS_FOR_FIELD[0]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_FIELDS_FOR_FIELD[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_FIELDS_FOR_FIELD[2]);
    res
}

pub(crate) static DTRACE_PRINT_XFIELD_FOR_FIELD_PROLOGUE: [&str; 2] =
    ["pub fn dtrace_print_", "(&self, depth: i32, prefix: String) {"];
pub(crate) fn build_dtrace_print_xfield_prologue(field_name: String) -> String {
    let mut res = String::from(DTRACE_PRINT_XFIELD_FOR_FIELD_PROLOGUE[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELD_FOR_FIELD_PROLOGUE[1]);
    res
}

pub(crate) static DTRACE_PRINT_XFIELD_FOR_FIELD_MID: &str = "if depth == 0 { return; }";
pub(crate) fn build_dtrace_print_xfield_middle() -> String {
    String::from(DTRACE_PRINT_XFIELD_FOR_FIELD_MID)
}

pub(crate) static DTRACE_PRINT_XFIELD_FOR_FIELD_EPILOGUE: &str = "}";
pub(crate) fn build_dtrace_print_xfield_epilogue() -> String {
    String::from(DTRACE_PRINT_XFIELD_FOR_FIELD_EPILOGUE)
}

pub(crate) static DTRACE_TMP_PRIM_VEC_FOR_FIELD: [&str; 11] = [
    "let mut __daikon_tmp",
    ": Vec<",
    "> = Vec::new(); let mut __daikon_tmp",
    " = 0; while __daikon_tmp",
    " < self.",
    ".len() { __daikon_tmp",
    ".push(",
    "::from_str(&self.",
    "[__daikon_tmp",
    "].to_string()).expect(\"Ok\")); __daikon_tmp",
    " += 1; }",
];
pub(crate) fn build_tmp_prim_vec_for_field(
    first_tmp: String,
    p_type: String,
    next_tmp: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_TMP_PRIM_VEC_FOR_FIELD[0]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[1]);
    res.push_str(&p_type);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[2]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[3]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[5]);
    res.push_str(&first_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[6]);
    res.push_str(&p_type);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[7]);
    res.push_str(&field_name);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[8]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[9]);
    res.push_str(&next_tmp);
    res.push_str(DTRACE_TMP_PRIM_VEC_FOR_FIELD[10]);
    res
}

pub(crate) static DTRACE_PRINT_PRIM_VEC_FOR_FIELD: [&str; 4] =
    ["dtrace_print_prim_vec::<", ">(&", ", format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_print_prim_vec_for_field(
    p_type: String,
    tmp_name: String,
    field_name: String,
) -> String {
    let mut res = String::from(DTRACE_PRINT_PRIM_VEC_FOR_FIELD[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRINT_PRIM_VEC_FOR_FIELD[1]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_PRIM_VEC_FOR_FIELD[2]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_PRIM_VEC_FOR_FIELD[3]);
    res
}

pub(crate) static DTRACE_PRINT_STRING_VEC_FOR_FIELD: [&str; 3] =
    ["dtrace_print_string_vec(&", ", format!(\"{}{}\", prefix, \".", "\"));"];
pub(crate) fn build_print_string_vec_for_field(tmp_name: String, field_name: String) -> String {
    let mut res = String::from(DTRACE_PRINT_STRING_VEC_FOR_FIELD[0]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_STRING_VEC_FOR_FIELD[1]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_STRING_VEC_FOR_FIELD[2]);
    res
}

pub(crate) static DTRACE_CALL_PRINT_FIELD: [&str; 2] =
    ["self.dtrace_print_", "(depth, prefix.clone());"];
pub(crate) fn build_call_print_field(field_name: String) -> String {
    let mut res = String::from(DTRACE_CALL_PRINT_FIELD[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_CALL_PRINT_FIELD[1]);
    res
}

#[allow(dead_code)]
pub(crate) static DTRACE_PRINT_XFIELDS_VEC: [&str; 6] = ["pub fn dtrace_print_",
                                                     "_vec(v: &Vec<&",
                                                        ">, var_name: String) {
                                                        let mut traces = match File::options().append(true).open(\"",
                                                        ".dtrace\") {
                                                            Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
                                                            Ok(traces) => traces,
                                                        };
                                                        writeln!(&mut traces, \"{}\", var_name).ok();
                                                        let mut arr = String::from(\"[\");
                                                        let mut i = 0;
                                                        while i+1 < v.len() {
                                                    arr.push_str(&format!(\"0x{:x} \", v[i].",
                                                        ".as_ptr() as usize));
                                                        i += 1;
                                                        }
                                                        if v.len() > 0 {
                                                    arr.push_str(&format!(\"0x{:x}\", v[i].",
                                                        ".as_ptr() as usize)); }
                                                        arr.push_str(\"]\");
                                                        writeln!(&mut traces, \"{}\", arr).ok();
                                                        writeln!(traces, \"0\").ok(); }"];
pub(crate) fn build_print_xfield_for_vec(field_name: String, basic_type: String) -> String {
    let mut res = String::from(DTRACE_PRINT_XFIELDS_VEC[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_VEC[1]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_PRINT_XFIELDS_VEC[2]);
    res.push_str(&*OUTPUT_NAME.lock().unwrap());
    res.push_str(DTRACE_PRINT_XFIELDS_VEC[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_VEC[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_VEC[5]);
    res
}

pub(crate) static LET_RET: [&str; 3] = ["fn __skip() { let __daikon_ret: ", " = ", "; }"];
pub(crate) fn build_let_ret(ret_ty: String, expr: String) -> String {
    let mut res = String::from(LET_RET[0]);
    res.push_str(&ret_ty);
    res.push_str(LET_RET[1]);
    res.push_str(&expr);
    res.push_str(LET_RET[2]);
    res
}

pub(crate) static RET: [&str; 1] = ["fn __skip() { return __daikon_ret; }"];
pub(crate) fn build_ret() -> String {
    String::from(RET[0])
}

// you have to delete this?
// make this an array with DTRACE_PRINT_FIELDS_EPILOGUE...
pub(crate) static DTRACE_PRINT_FIELDS_PROLOGUE: &str = "impl __skip { pub fn dtrace_print_fields(&self, depth: i32, prefix: String) { if depth == 0 { return; } ";
pub(crate) fn dtrace_print_fields_prologue() -> String {
    String::from(DTRACE_PRINT_FIELDS_PROLOGUE)
}

pub(crate) static DTRACE_PRINT_FIELDS_EPILOGUE: &str = "} } struct __skip{}"; // maybe can avoid deleting it, but still bad
pub(crate) fn dtrace_print_fields_epilogue() -> String {
    String::from(DTRACE_PRINT_FIELDS_EPILOGUE)
}

pub(crate) static DTRACE_PRINT_FIELDS_VEC_PROLOGUE: [&str; 2] = [
    "impl __skip { pub fn dtrace_print_fields_vec(v: &Vec<&",
    ">, depth: i32, prefix: String) { if depth == 0 { return; } ",
];
pub(crate) fn dtrace_print_fields_vec_prologue(spliced_struct: String) -> String {
    let mut res = String::from(DTRACE_PRINT_FIELDS_VEC_PROLOGUE[0]);
    res.push_str(&spliced_struct);
    res.push_str(DTRACE_PRINT_FIELDS_VEC_PROLOGUE[1]);
    res
}

pub(crate) static DTRACE_PRINT_FIELDS_VEC_EPILOGUE: &str = "} } struct __skip{}";
pub(crate) fn dtrace_print_fields_vec_epilogue() -> String {
    String::from(DTRACE_PRINT_FIELDS_VEC_EPILOGUE)
}

pub(crate) static DTRACE_PRINT_XFIELDS_VEC_PROLOGUE: &str = "impl __skip {";
pub(crate) fn dtrace_print_xfields_vec_prologue() -> String {
    String::from(DTRACE_PRINT_XFIELDS_VEC_PROLOGUE)
}

pub(crate) static DTRACE_PRINT_XFIELDS_VEC_EPILOGUE: &str = " }";
pub(crate) fn dtrace_print_xfields_vec_epilogue() -> String {
    String::from(DTRACE_PRINT_XFIELDS_VEC_EPILOGUE)
}

pub(crate) static DTRACE_PRINT_XFIELDS: [&str; 6] = ["pub fn dtrace_print_",
                                                     "_vec(v: &Vec<&",
                                                        ">, var_name: String) {
                                                        let mut traces = match File::options().append(true).open(\"",
                                                        ".dtrace\") {
                                                            Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
                                                            Ok(traces) => traces,
                                                        };
                                                        writeln!(&mut traces, \"{}\", var_name).ok();
                                                        let mut arr = String::from(\"[\");
                                                        let mut i = 0;
                                                        while i+1 < v.len() {
                                                    arr.push_str(&format!(\"{} \", v[i].",
                                                        "));
                                                        i += 1;
                                                        }
                                                        if v.len() > 0 {
                                                    arr.push_str(&format!(\"{}\", v[i].",
                                                        ")); }
                                                        arr.push_str(\"]\");
                                                        writeln!(&mut traces, \"{}\", arr).ok();
                                                        writeln!(traces, \"0\").ok(); }"];
pub(crate) fn build_print_xfield(field_name: String, basic_type: String) -> String {
    let mut res = String::from(DTRACE_PRINT_XFIELDS[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS[1]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_PRINT_XFIELDS[2]);
    res.push_str(&*OUTPUT_NAME.lock().unwrap());
    res.push_str(DTRACE_PRINT_XFIELDS[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS[5]);
    res
}

pub(crate) static DTRACE_PRINT_XFIELDS_STRING: [&str; 6] = ["pub fn dtrace_print_",
                                                     "_vec(v: &Vec<&",
                                                        ">, var_name: String) {
                                                        let mut traces = match File::options().append(true).open(\"",
                                                        ".dtrace\") {
                                                            Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
                                                            Ok(traces) => traces,
                                                        };
                                                        writeln!(&mut traces, \"{}\", var_name).ok();
                                                        let mut arr = String::from(\"[\");
                                                        let mut i = 0;
                                                        while i+1 < v.len() {
                                                    arr.push_str(&format!(\"\\\"{}\\\" \", v[i].",
                                                        "));
                                                        i += 1;
                                                        }
                                                        if v.len() > 0 {
                                                    arr.push_str(&format!(\"\\\"{}\\\"\", v[i].",
                                                        ")); }
                                                        arr.push_str(\"]\");
                                                        writeln!(&mut traces, \"{}\", arr).ok();
                                                        writeln!(traces, \"0\").ok(); }"];
pub(crate) fn build_print_xfield_string(field_name: String, basic_type: String) -> String {
    let mut res = String::from(DTRACE_PRINT_XFIELDS_STRING[0]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_STRING[1]);
    res.push_str(&basic_type);
    res.push_str(DTRACE_PRINT_XFIELDS_STRING[2]);
    res.push_str(&*OUTPUT_NAME.lock().unwrap());
    res.push_str(DTRACE_PRINT_XFIELDS_STRING[3]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_STRING[4]);
    res.push_str(&field_name);
    res.push_str(DTRACE_PRINT_XFIELDS_STRING[5]);
    res
}

pub(crate) static BUILD_POINTER_ARR: [&str; 3] =
    ["dtrace_print_pointer(", " as *const _ as *const () as usize, String::from(\"", "\"));"];
pub(crate) fn build_pointer_arr(var_name: String) -> String {
    let mut res = String::from(BUILD_POINTER_ARR[0]);
    res.push_str(&var_name);
    res.push_str(BUILD_POINTER_ARR[1]);
    res.push_str(&var_name);
    res.push_str(BUILD_POINTER_ARR[2]);
    res
}

// does this work for references?
pub(crate) static BUILD_POINTER_ARR_RET: &str = "dtrace_print_pointer(__daikon_ret as *const _ as *const () as usize, String::from(\"return\"));";
pub(crate) fn build_pointer_arr_ret() -> String {
    String::from(BUILD_POINTER_ARR_RET)
}

pub(crate) static DTRACE_PRINT_FIELDS_NOOP: &str =
    "pub fn dtrace_print_fields(&self, _depth: i32, _prefix: String) {}";
pub(crate) fn build_dtrace_print_fields_noop() -> String {
    String::from(DTRACE_PRINT_FIELDS_NOOP)
}

pub(crate) static DTRACE_PRINT_FIELDS_VEC_NOOP: [&str; 2] =
    ["pub fn dtrace_print_fields_vec(_v: &Vec<&", ">, _depth: i32, _prefix: String) {}"];
pub(crate) fn build_dtrace_print_fields_vec_noop(basic_struct: String) -> String {
    let mut res = String::from(DTRACE_PRINT_FIELDS_VEC_NOOP[0]);
    res.push_str(&basic_struct);
    res.push_str(DTRACE_PRINT_FIELDS_VEC_NOOP[1]);
    res
}

// only end fn __skip because we will smash a tmp vec loop on the front.
pub(crate) static DTRACE_PRINT_PRIM_VEC: [&str; 4] =
    ["dtrace_print_prim_vec::<", ">(&", ", String::from(\"", "\")); }"];
pub(crate) fn build_print_prim_vec(p_type: String, tmp_name: String, var_name: String) -> String {
    let mut res = String::from(DTRACE_PRINT_PRIM_VEC[0]);
    res.push_str(&p_type);
    res.push_str(DTRACE_PRINT_PRIM_VEC[1]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_PRIM_VEC[2]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRINT_PRIM_VEC[3]);
    res
}

pub(crate) static DTRACE_PRINT_STRING_VEC: [&str; 3] =
    ["dtrace_print_string_vec(&", ", String::from(\"", "\")); }"];
pub(crate) fn build_print_string_vec(tmp_name: String, var_name: String) -> String {
    let mut res = String::from(DTRACE_PRINT_STRING_VEC[0]);
    res.push_str(&tmp_name);
    res.push_str(DTRACE_PRINT_STRING_VEC[1]);
    res.push_str(&var_name);
    res.push_str(DTRACE_PRINT_STRING_VEC[2]);
    res
}

pub(crate) static BUILD_A_IMPL_BLOCK: &str = "impl __skip {}";
pub(crate) fn base_impl() -> String {
    String::from(BUILD_A_IMPL_BLOCK)
}

pub(crate) static FABRICATE_TYPE_FOR_IMPL: [&str; 3] = ["fn __skip() -> ", " {}\nstruct ", "{}"];
pub(crate) fn build_phony_ret(struct_name: String) -> String {
    let mut res = String::from(FABRICATE_TYPE_FOR_IMPL[0]);
    res.push_str(&struct_name);
    res.push_str(FABRICATE_TYPE_FOR_IMPL[1]);
    res.push_str(&struct_name);
    res.push_str(FABRICATE_TYPE_FOR_IMPL[2]);
    res
}

pub(crate) static VOID_RETURN: &str = "fn __skip() { return; }";
pub(crate) fn build_void_return() -> String {
    String::from(VOID_RETURN)
}

pub(crate) static DTRACE_NEWLINE: &str = "fn __skip() { dtrace_newline(); }";
pub(crate) fn dtrace_newline() -> String {
    String::from(DTRACE_NEWLINE)
}

// this NONCE_COUNTER per-file is broken for multi-file non-concurrent programs. It has to be a single counter shared between all the files.
// Difficult in Rust as there is no easy extern escape like in C. Maybe unsafe.
pub(crate) static IMPORTS: &str = "use std::fs::File;\nuse std::io::prelude::*;\nuse std::sync::{LazyLock, Mutex};\nuse std::str::FromStr;\nstatic NONCE_COUNTER: LazyLock<Mutex<u32>> = LazyLock::new(|| Mutex::new(0));";
pub(crate) fn build_imports() -> String {
    String::from(IMPORTS)
}

pub(crate) static DAIKON_LIB: [&str; 15] = [
    "pub fn dtrace_print_pointer_arr<T>(v: &[&T], var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    let mut arr = String::from(\"[\");
    let mut i = 0;
    while i+1 < v.len() {
        arr.push_str(&format!(\"0x{:x} \", v[i] as *const _ as usize));
        i += 1;
    }
    if v.len() > 0 {
        arr.push_str(&format!(\"0x{:x}\", v[i] as *const _ as usize));
    }
    arr.push_str(\"]\");
    writeln!(&mut traces, \"{}\", arr).ok();
    writeln!(&mut traces, \"0\").ok();
}

pub fn dtrace_print_pointer_vec<T>(v: &Vec<&T>, var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    let mut arr = String::from(\"[\");
    let mut i = 0;
    while i+1 < v.len() {
        arr.push_str(&format!(\"0x{:x} \", v[i] as *const _ as usize));
        i += 1;
    }
    if v.len() > 0 {
        arr.push_str(&format!(\"0x{:x}\", v[i] as *const _ as usize));
    }
    arr.push_str(\"]\");
    writeln!(&mut traces, \"{}\", arr).ok();
    writeln!(&mut traces, \"0\").ok();
}

// T must implement Display trait
fn dtrace_print_prim_arr<T: std::fmt::Display>(v: &[T], prefix: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", format!(\"{}{}\", prefix, \"[..]\")).ok();
    let mut arr = String::from(\"[\");
    let mut i = 0;
    while i+1 < v.len() {
        arr.push_str(&format!(\"{} \", v[i]));
        i += 1;
    }
    if v.len() > 0 {
        arr.push_str(&format!(\"{}\", v[i]));
    }
    arr.push_str(\"]\");
    writeln!(&mut traces, \"{}\", arr).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_print_prim_vec<T: std::fmt::Display>(v: &Vec<T>, prefix: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", format!(\"{}{}\", prefix, \"[..]\")).ok();
    let mut arr = String::from(\"[\");
    let mut i = 0;
    while i+1 < v.len() {
        arr.push_str(&format!(\"{} \", v[i]));
        i += 1;
    }
    if v.len() > 0 {
        arr.push_str(&format!(\"{}\", v[i]));
    }
    arr.push_str(\"]\");
    writeln!(&mut traces, \"{}\", arr).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_print_str(v: &str, var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    writeln!(&mut traces, \"{}\", v).ok();
    writeln!(&mut traces, \"0\").ok();
}

// T must implement Display trait
fn dtrace_print_prim<T: std::fmt::Display>(v: T, var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    writeln!(&mut traces, \"{}\", v).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_print_string(v: String, var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    writeln!(&mut traces, \"\\\"{}\\\"\", v).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_print_string_vec(v: &Vec<String>, prefix: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", format!(\"{}{}\", prefix, \"[..]\")).ok();
    let mut arr = String::from(\"[\");
    let mut i = 0;
    while i+1 < v.len() {
        arr.push_str(&format!(\"\\\"{}\\\" \", v[i]));
        i += 1;
    }
    if v.len() > 0 {
        arr.push_str(&format!(\"\\\"{}\\\"\", v[i]));
    }
    arr.push_str(\"]\");
    writeln!(&mut traces, \"{}\", arr).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_print_pointer(v: usize, var_name: String) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", var_name).ok();
    writeln!(&mut traces, \"0x{:x}\", v).ok();
    writeln!(&mut traces, \"0\").ok();
}

fn dtrace_entry_no_nonce(ppt_name: &str) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", ppt_name).ok();
}

fn dtrace_exit_no_nonce(ppt_name: &str) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", ppt_name).ok();
}

fn dtrace_entry(ppt_name: &str, nonce: u32) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(&mut traces, \"{}\", ppt_name).ok();
    writeln!(&mut traces, \"this_invocation_nonce\").ok();
    writeln!(&mut traces, \"{}\", nonce).ok();
}

fn dtrace_exit(ppt_name: &str, nonce: u32) {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(traces, \"{}\", ppt_name).ok();
    writeln!(traces, \"this_invocation_nonce\").ok();
    writeln!(traces, \"{}\", nonce).ok();
}

fn dtrace_newline() {
    let mut traces = match File::options().append(true).open(\"",
    ".dtrace\") {
        Err(why) => panic!(\"Daikon couldn't open file, {}\", why),
        Ok(traces) => traces,
    };
    writeln!(traces, \"\").ok();
}",
];

pub(crate) fn daikon_lib() -> String {
    let mut res = String::from(DAIKON_LIB[0]);
    for i in 1..DAIKON_LIB.len() {
        res.push_str(&*OUTPUT_NAME.lock().unwrap());
        res.push_str(DAIKON_LIB[i]);
    }
    res
}
