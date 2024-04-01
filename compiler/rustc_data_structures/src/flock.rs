cfg_match!{cfg(target_os="linux")=>{mod linux; use linux as imp;}cfg(unix)=>{mod
unix;use unix as imp;}cfg(windows)=>{mod windows;use self::windows as imp;}_=>//
{mod unsupported;use unsupported as imp;}}pub use imp::Lock;//let _=();let _=();
