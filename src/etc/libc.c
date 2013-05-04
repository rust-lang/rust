/*
 * This calculates the platform-variable portion of the libc module.
 * Move code in here only as you discover it is platform-variable.
 *
 */

 /* c95 */
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <wchar.h>

/* c99 */
#include <inttypes.h>

/* posix */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define S(T) ((((T)-1)<0) ? 'i' : 'u')
#define B(T) (((int)sizeof(T)) * CHAR_BIT)
#define put_type(N,T) \
        printf("        type %s = %c%d;\n", N, S(T), B(T))

#define put_ftype(N,T) \
        printf("        type %s = f%d;\n", N, B(T))

#define CT(T) ((((T)-1)<0) ? "int" : "uint")
#define CS(T) ((((T)-1)<0) ? "" : "_u")
#define put_const(N,T)                            \
        printf("        const %s : %s = %d%s;\n", \
               #N, CT(T), N, CS(T))

void c95_types() {
  printf("    mod c95 {\n");

  put_type("c_char", char);
  put_type("c_schar", signed char);
  put_type("c_uchar", unsigned char);

  put_type("c_short", short);
  put_type("c_ushort", unsigned short);

  put_type("c_int", int);
  put_type("c_uint", unsigned int);

  put_type("c_long", long);
  put_type("c_ulong", unsigned long);

  put_ftype("c_float", float);
  put_ftype("c_double", double);

  put_type("size_t", size_t);
  put_type("ptrdiff_t", ptrdiff_t);

  put_type("clock_t", clock_t);
  put_type("time_t", time_t);

  put_type("wchar_t", wchar_t);

  printf("    }\n");
}

void c99_types() {
  printf("    mod c99 {\n");

  put_type("c_longlong", long long);
  put_type("c_ulonglong", unsigned long long);

  put_type("intptr_t", intptr_t);
  put_type("uintptr_t", uintptr_t);

  printf("    }\n");
}

void posix88_types() {
  printf("    mod posix88 {\n");

  put_type("off_t", off_t);
  put_type("dev_t", dev_t);
  put_type("ino_t", ino_t);
  put_type("pid_t", pid_t);
#ifndef __WIN32__
  put_type("uid_t", uid_t);
  put_type("gid_t", gid_t);
#endif
  put_type("useconds_t", useconds_t);
  put_type("mode_t", mode_t);

  put_type("ssize_t", ssize_t);

  printf("    }\n");
}

void extra_types() {
  printf("    mod extra {\n");
  printf("    }\n");
}


void c95_consts() {
  printf("    mod c95 {\n");

  put_const(EXIT_FAILURE, int);
  put_const(EXIT_SUCCESS, int);
  put_const(RAND_MAX, int);

  put_const(EOF, int);
  put_const(SEEK_SET, int);
  put_const(SEEK_CUR, int);
  put_const(SEEK_END, int);

  put_const(_IOFBF, int);
  put_const(_IONBF, int);
  put_const(_IOLBF, int);

  put_const(BUFSIZ, size_t);
  put_const(FOPEN_MAX, size_t);
  put_const(FILENAME_MAX, size_t);
  put_const(L_tmpnam, size_t);
  put_const(TMP_MAX, size_t);

  printf("    }\n");
}


void posix88_consts() {
  printf("    mod posix88 {\n");
  put_const(O_RDONLY, int);
  put_const(O_WRONLY, int);
  put_const(O_RDWR, int);
  put_const(O_APPEND, int);
  put_const(O_CREAT, int);
  put_const(O_EXCL, int);
  put_const(O_TRUNC, int);

  put_const(S_IFIFO, int);
  put_const(S_IFCHR, int);
  put_const(S_IFBLK, int);
  put_const(S_IFDIR, int);
  put_const(S_IFREG, int);
  put_const(S_IFMT, int);

  put_const(S_IEXEC, int);
  put_const(S_IWRITE, int);
  put_const(S_IREAD, int);

  put_const(S_IRWXU, int);
  put_const(S_IXUSR, int);
  put_const(S_IWUSR, int);
  put_const(S_IRUSR, int);

#ifdef F_OK
  put_const(F_OK, int);
#endif
#ifdef R_OK
  put_const(R_OK, int);
#endif
#ifdef W_OK
  put_const(W_OK, int);
#endif
#ifdef X_OK
  put_const(X_OK, int);
#endif

#ifdef STDIN_FILENO
  put_const(STDIN_FILENO, int);
#endif
#ifdef STDOUT_FILENO
  put_const(STDOUT_FILENO, int);
#endif
#ifdef STDERR_FILENO
  put_const(STDERR_FILENO, int);
#endif

#ifdef F_LOCK
  put_const(F_LOCK, int);
#endif

#ifdef F_TEST
  put_const(F_TEST, int);
#endif

#ifdef F_TLOCK
  put_const(F_TLOCK, int);
#endif

#ifdef F_ULOCK
  put_const(F_ULOCK, int);
#endif

  printf("    }\n");
}

void extra_consts() {
  printf("    mod extra {\n");
#ifdef O_RSYNC
  put_const(O_RSYNC, int);
#endif

#ifdef O_DSYNC
  put_const(O_DSYNC, int);
#endif

#ifdef O_SYNC
  put_const(O_SYNC, int);
#endif

#ifdef O_TEXT
  put_const(O_TEXT, int);
#endif

#ifdef O_BINARY
  put_const(O_BINARY, int);
#endif

#ifdef O_IRUSR
  put_const(O_IRUSR, int);
#endif

#ifdef O_IWUSR
  put_const(O_IWUSR, int);
#endif

  printf("    }\n");
}

int main() {
  printf("mod types {");
  c95_types();
  c99_types();
  posix88_types();
  extra_types();
  printf("}\n");

  printf("mod consts {\n");
  c95_consts();
  posix88_consts();
  extra_consts();
  printf("}\n");
}
