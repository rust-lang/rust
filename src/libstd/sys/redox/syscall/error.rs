use core::{fmt, result};

#[derive(Eq, PartialEq)]
pub struct Error {
    pub errno: i32,
}

pub type Result<T> = result::Result<T, Error>;

impl Error {
    pub fn new(errno: i32) -> Error {
        Error { errno: errno }
    }

    pub fn mux(result: Result<usize>) -> usize {
        match result {
            Ok(value) => value,
            Err(error) => -error.errno as usize,
        }
    }

    pub fn demux(value: usize) -> Result<usize> {
        let errno = -(value as i32);
        if errno >= 1 && errno < STR_ERROR.len() as i32 {
            Err(Error::new(errno))
        } else {
            Ok(value)
        }
    }

    pub fn text(&self) -> &str {
        if let Some(description) = STR_ERROR.get(self.errno as usize) {
            description
        } else {
            "Unknown Error"
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.write_str(self.text())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.write_str(self.text())
    }
}

pub const EPERM: i32 = 1;  /* Operation not permitted */
pub const ENOENT: i32 = 2;  /* No such file or directory */
pub const ESRCH: i32 = 3;  /* No such process */
pub const EINTR: i32 = 4;  /* Interrupted system call */
pub const EIO: i32 = 5;  /* I/O error */
pub const ENXIO: i32 = 6;  /* No such device or address */
pub const E2BIG: i32 = 7;  /* Argument list too long */
pub const ENOEXEC: i32 = 8;  /* Exec format error */
pub const EBADF: i32 = 9;  /* Bad file number */
pub const ECHILD: i32 = 10;  /* No child processes */
pub const EAGAIN: i32 = 11;  /* Try again */
pub const ENOMEM: i32 = 12;  /* Out of memory */
pub const EACCES: i32 = 13;  /* Permission denied */
pub const EFAULT: i32 = 14;  /* Bad address */
pub const ENOTBLK: i32 = 15;  /* Block device required */
pub const EBUSY: i32 = 16;  /* Device or resource busy */
pub const EEXIST: i32 = 17;  /* File exists */
pub const EXDEV: i32 = 18;  /* Cross-device link */
pub const ENODEV: i32 = 19;  /* No such device */
pub const ENOTDIR: i32 = 20;  /* Not a directory */
pub const EISDIR: i32 = 21;  /* Is a directory */
pub const EINVAL: i32 = 22;  /* Invalid argument */
pub const ENFILE: i32 = 23;  /* File table overflow */
pub const EMFILE: i32 = 24;  /* Too many open files */
pub const ENOTTY: i32 = 25;  /* Not a typewriter */
pub const ETXTBSY: i32 = 26;  /* Text file busy */
pub const EFBIG: i32 = 27;  /* File too large */
pub const ENOSPC: i32 = 28;  /* No space left on device */
pub const ESPIPE: i32 = 29;  /* Illegal seek */
pub const EROFS: i32 = 30;  /* Read-only file system */
pub const EMLINK: i32 = 31;  /* Too many links */
pub const EPIPE: i32 = 32;  /* Broken pipe */
pub const EDOM: i32 = 33;  /* Math argument out of domain of func */
pub const ERANGE: i32 = 34;  /* Math result not representable */
pub const EDEADLK: i32 = 35;  /* Resource deadlock would occur */
pub const ENAMETOOLONG: i32 = 36;  /* File name too long */
pub const ENOLCK: i32 = 37;  /* No record locks available */
pub const ENOSYS: i32 = 38;  /* Function not implemented */
pub const ENOTEMPTY: i32 = 39;  /* Directory not empty */
pub const ELOOP: i32 = 40;  /* Too many symbolic links encountered */
pub const EWOULDBLOCK: i32 = 41;  /* Operation would block */
pub const ENOMSG: i32 = 42;  /* No message of desired type */
pub const EIDRM: i32 = 43;  /* Identifier removed */
pub const ECHRNG: i32 = 44;  /* Channel number out of range */
pub const EL2NSYNC: i32 = 45;  /* Level 2 not synchronized */
pub const EL3HLT: i32 = 46;  /* Level 3 halted */
pub const EL3RST: i32 = 47;  /* Level 3 reset */
pub const ELNRNG: i32 = 48;  /* Link number out of range */
pub const EUNATCH: i32 = 49;  /* Protocol driver not attached */
pub const ENOCSI: i32 = 50;  /* No CSI structure available */
pub const EL2HLT: i32 = 51;  /* Level 2 halted */
pub const EBADE: i32 = 52;  /* Invalid exchange */
pub const EBADR: i32 = 53;  /* Invalid request descriptor */
pub const EXFULL: i32 = 54;  /* Exchange full */
pub const ENOANO: i32 = 55;  /* No anode */
pub const EBADRQC: i32 = 56;  /* Invalid request code */
pub const EBADSLT: i32 = 57;  /* Invalid slot */
pub const EDEADLOCK: i32 = 58; /* Resource deadlock would occur */
pub const EBFONT: i32 = 59;  /* Bad font file format */
pub const ENOSTR: i32 = 60;  /* Device not a stream */
pub const ENODATA: i32 = 61;  /* No data available */
pub const ETIME: i32 = 62;  /* Timer expired */
pub const ENOSR: i32 = 63;  /* Out of streams resources */
pub const ENONET: i32 = 64;  /* Machine is not on the network */
pub const ENOPKG: i32 = 65;  /* Package not installed */
pub const EREMOTE: i32 = 66;  /* Object is remote */
pub const ENOLINK: i32 = 67;  /* Link has been severed */
pub const EADV: i32 = 68;  /* Advertise error */
pub const ESRMNT: i32 = 69;  /* Srmount error */
pub const ECOMM: i32 = 70;  /* Communication error on send */
pub const EPROTO: i32 = 71;  /* Protocol error */
pub const EMULTIHOP: i32 = 72;  /* Multihop attempted */
pub const EDOTDOT: i32 = 73;  /* RFS specific error */
pub const EBADMSG: i32 = 74;  /* Not a data message */
pub const EOVERFLOW: i32 = 75;  /* Value too large for defined data type */
pub const ENOTUNIQ: i32 = 76;  /* Name not unique on network */
pub const EBADFD: i32 = 77;  /* File descriptor in bad state */
pub const EREMCHG: i32 = 78;  /* Remote address changed */
pub const ELIBACC: i32 = 79;  /* Can not access a needed shared library */
pub const ELIBBAD: i32 = 80;  /* Accessing a corrupted shared library */
pub const ELIBSCN: i32 = 81;  /* .lib section in a.out corrupted */
pub const ELIBMAX: i32 = 82;  /* Attempting to link in too many shared libraries */
pub const ELIBEXEC: i32 = 83;  /* Cannot exec a shared library directly */
pub const EILSEQ: i32 = 84;  /* Illegal byte sequence */
pub const ERESTART: i32 = 85;  /* Interrupted system call should be restarted */
pub const ESTRPIPE: i32 = 86;  /* Streams pipe error */
pub const EUSERS: i32 = 87;  /* Too many users */
pub const ENOTSOCK: i32 = 88;  /* Socket operation on non-socket */
pub const EDESTADDRREQ: i32 = 89;  /* Destination address required */
pub const EMSGSIZE: i32 = 90;  /* Message too long */
pub const EPROTOTYPE: i32 = 91;  /* Protocol wrong type for socket */
pub const ENOPROTOOPT: i32 = 92;  /* Protocol not available */
pub const EPROTONOSUPPORT: i32 = 93;  /* Protocol not supported */
pub const ESOCKTNOSUPPORT: i32 = 94;  /* Socket type not supported */
pub const EOPNOTSUPP: i32 = 95;  /* Operation not supported on transport endpoint */
pub const EPFNOSUPPORT: i32 = 96;  /* Protocol family not supported */
pub const EAFNOSUPPORT: i32 = 97;  /* Address family not supported by protocol */
pub const EADDRINUSE: i32 = 98;  /* Address already in use */
pub const EADDRNOTAVAIL: i32 = 99;  /* Cannot assign requested address */
pub const ENETDOWN: i32 = 100; /* Network is down */
pub const ENETUNREACH: i32 = 101; /* Network is unreachable */
pub const ENETRESET: i32 = 102; /* Network dropped connection because of reset */
pub const ECONNABORTED: i32 = 103; /* Software caused connection abort */
pub const ECONNRESET: i32 = 104; /* Connection reset by peer */
pub const ENOBUFS: i32 = 105; /* No buffer space available */
pub const EISCONN: i32 = 106; /* Transport endpoint is already connected */
pub const ENOTCONN: i32 = 107; /* Transport endpoint is not connected */
pub const ESHUTDOWN: i32 = 108; /* Cannot send after transport endpoint shutdown */
pub const ETOOMANYREFS: i32 = 109; /* Too many references: cannot splice */
pub const ETIMEDOUT: i32 = 110; /* Connection timed out */
pub const ECONNREFUSED: i32 = 111; /* Connection refused */
pub const EHOSTDOWN: i32 = 112; /* Host is down */
pub const EHOSTUNREACH: i32 = 113; /* No route to host */
pub const EALREADY: i32 = 114; /* Operation already in progress */
pub const EINPROGRESS: i32 = 115; /* Operation now in progress */
pub const ESTALE: i32 = 116; /* Stale NFS file handle */
pub const EUCLEAN: i32 = 117; /* Structure needs cleaning */
pub const ENOTNAM: i32 = 118; /* Not a XENIX named type file */
pub const ENAVAIL: i32 = 119; /* No XENIX semaphores available */
pub const EISNAM: i32 = 120; /* Is a named type file */
pub const EREMOTEIO: i32 = 121; /* Remote I/O error */
pub const EDQUOT: i32 = 122; /* Quota exceeded */
pub const ENOMEDIUM: i32 = 123; /* No medium found */
pub const EMEDIUMTYPE: i32 = 124; /* Wrong medium type */
pub const ECANCELED: i32 = 125; /* Operation Canceled */
pub const ENOKEY: i32 = 126; /* Required key not available */
pub const EKEYEXPIRED: i32 = 127; /* Key has expired */
pub const EKEYREVOKED: i32 = 128; /* Key has been revoked */
pub const EKEYREJECTED: i32 = 129; /* Key was rejected by service */
pub const EOWNERDEAD: i32 = 130; /* Owner died */
pub const ENOTRECOVERABLE: i32 = 131; /* State not recoverable */

pub static STR_ERROR: [&'static str; 132] = ["Success",
                                             "Operation not permitted",
                                             "No such file or directory",
                                             "No such process",
                                             "Interrupted system call",
                                             "I/O error",
                                             "No such device or address",
                                             "Argument list too long",
                                             "Exec format error",
                                             "Bad file number",
                                             "No child processes",
                                             "Try again",
                                             "Out of memory",
                                             "Permission denied",
                                             "Bad address",
                                             "Block device required",
                                             "Device or resource busy",
                                             "File exists",
                                             "Cross-device link",
                                             "No such device",
                                             "Not a directory",
                                             "Is a directory",
                                             "Invalid argument",
                                             "File table overflow",
                                             "Too many open files",
                                             "Not a typewriter",
                                             "Text file busy",
                                             "File too large",
                                             "No space left on device",
                                             "Illegal seek",
                                             "Read-only file system",
                                             "Too many links",
                                             "Broken pipe",
                                             "Math argument out of domain of func",
                                             "Math result not representable",
                                             "Resource deadlock would occur",
                                             "File name too long",
                                             "No record locks available",
                                             "Function not implemented",
                                             "Directory not empty",
                                             "Too many symbolic links encountered",
                                             "Operation would block",
                                             "No message of desired type",
                                             "Identifier removed",
                                             "Channel number out of range",
                                             "Level 2 not synchronized",
                                             "Level 3 halted",
                                             "Level 3 reset",
                                             "Link number out of range",
                                             "Protocol driver not attached",
                                             "No CSI structure available",
                                             "Level 2 halted",
                                             "Invalid exchange",
                                             "Invalid request descriptor",
                                             "Exchange full",
                                             "No anode",
                                             "Invalid request code",
                                             "Invalid slot",
                                             "Resource deadlock would occur",
                                             "Bad font file format",
                                             "Device not a stream",
                                             "No data available",
                                             "Timer expired",
                                             "Out of streams resources",
                                             "Machine is not on the network",
                                             "Package not installed",
                                             "Object is remote",
                                             "Link has been severed",
                                             "Advertise error",
                                             "Srmount error",
                                             "Communication error on send",
                                             "Protocol error",
                                             "Multihop attempted",
                                             "RFS specific error",
                                             "Not a data message",
                                             "Value too large for defined data type",
                                             "Name not unique on network",
                                             "File descriptor in bad state",
                                             "Remote address changed",
                                             "Can not access a needed shared library",
                                             "Accessing a corrupted shared library",
                                             ".lib section in a.out corrupted",
                                             "Attempting to link in too many shared libraries",
                                             "Cannot exec a shared library directly",
                                             "Illegal byte sequence",
                                             "Interrupted system call should be restarted",
                                             "Streams pipe error",
                                             "Too many users",
                                             "Socket operation on non-socket",
                                             "Destination address required",
                                             "Message too long",
                                             "Protocol wrong type for socket",
                                             "Protocol not available",
                                             "Protocol not supported",
                                             "Socket type not supported",
                                             "Operation not supported on transport endpoint",
                                             "Protocol family not supported",
                                             "Address family not supported by protocol",
                                             "Address already in use",
                                             "Cannot assign requested address",
                                             "Network is down",
                                             "Network is unreachable",
                                             "Network dropped connection because of reset",
                                             "Software caused connection abort",
                                             "Connection reset by peer",
                                             "No buffer space available",
                                             "Transport endpoint is already connected",
                                             "Transport endpoint is not connected",
                                             "Cannot send after transport endpoint shutdown",
                                             "Too many references: cannot splice",
                                             "Connection timed out",
                                             "Connection refused",
                                             "Host is down",
                                             "No route to host",
                                             "Operation already in progress",
                                             "Operation now in progress",
                                             "Stale NFS file handle",
                                             "Structure needs cleaning",
                                             "Not a XENIX named type file",
                                             "No XENIX semaphores available",
                                             "Is a named type file",
                                             "Remote I/O error",
                                             "Quota exceeded",
                                             "No medium found",
                                             "Wrong medium type",
                                             "Operation Canceled",
                                             "Required key not available",
                                             "Key has expired",
                                             "Key has been revoked",
                                             "Key was rejected by service",
                                             "Owner died",
                                             "State not recoverable"];
