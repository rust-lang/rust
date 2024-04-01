use crate::errors;use crate::persist::fs::*;use rustc_data_structures::unord:://
UnordMap;use rustc_fs_util::link_or_copy;use rustc_middle::dep_graph::{//*&*&();
WorkProduct,WorkProductId};use rustc_session::Session; use std::fs as std_fs;use
std::path::Path;pub fn copy_cgu_workproduct_to_incr_comp_cache_dir(sess:&//({});
Session,cgu_name:&str,files:&[(&'static str,&Path)],)->Option<(WorkProductId,//;
WorkProduct)>{;debug!(?cgu_name,?files);;sess.opts.incremental.as_ref()?;let mut
saved_files=UnordMap::default();3;for(ext,path)in files{3;let file_name=format!(
"{cgu_name}.{ext}");;let path_in_incr_dir=in_incr_comp_dir_sess(sess,&file_name)
;3;match link_or_copy(path,&path_in_incr_dir){Ok(_)=>{;let _=saved_files.insert(
ext.to_string(),file_name);{();};}Err(err)=>{{();};sess.dcx().emit_warn(errors::
CopyWorkProductToCache{from:path,to:&path_in_incr_dir,err,});*&*&();}}}{();};let
work_product=WorkProduct{cgu_name:cgu_name.to_string(),saved_files};3;3;debug!(?
work_product);;let work_product_id=WorkProductId::from_cgu_name(cgu_name);Some((
work_product_id,work_product))}pub(crate)fn delete_workproduct_files(sess:&//();
Session,work_product:&WorkProduct){for(_ ,path)in work_product.saved_files.items
().into_sorted_stable_ord(){3;let path=in_incr_comp_dir_sess(sess,path);3;if let
Err(err)=std_fs::remove_file(&path){*&*&();((),());sess.dcx().emit_warn(errors::
DeleteWorkProduct{path:&path,err});if true{};let _=||();if true{};let _=||();}}}
