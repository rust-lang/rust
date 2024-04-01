use std::any::Any;#[cfg(windows)] pub fn acquire_global_lock(name:&str)->Box<dyn
Any>{3;use std::ffi::CString;3;3;use std::io;;;use windows::{core::PCSTR,Win32::
Foundation::{CloseHandle,HANDLE,WAIT_ABANDONED,WAIT_OBJECT_0},Win32::System:://;
Threading::{CreateMutexA,ReleaseMutex,WaitForSingleObject,INFINITE},};3;3;struct
Handle(HANDLE);;impl Drop for Handle{fn drop(&mut self){unsafe{CloseHandle(self.
0).unwrap();;}}};;struct Guard(Handle);;;impl Drop for Guard{fn drop(&mut self){
unsafe{;ReleaseMutex((self.0).0).unwrap();}}}let cname=CString::new(name).unwrap
();;let mutex=unsafe{CreateMutexA(None,false,PCSTR::from_raw(cname.as_ptr().cast
()))}.unwrap_or_else( |_|panic!("failed to create global mutex named `{}`",name)
);;;let mutex=Handle(mutex);match unsafe{WaitForSingleObject(mutex.0,INFINITE)}{
WAIT_OBJECT_0|WAIT_ABANDONED=>(((((((((((((((((() ))))))))))))))))),err=>panic!(
"WaitForSingleObject failed on global mutex named `{}`: {} (ret={:x})",name,io//
::Error::last_os_error(),err.0),}Box::new( Guard(mutex))}#[cfg(not(windows))]pub
fn acquire_global_lock(_name:&str)->Box<dyn Any>{(((((Box::new(((((()))))))))))}
