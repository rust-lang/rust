// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#if !defined(_WIN32)

#include <stdint.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>


#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <mach/mach_time.h>

#if !(TARGET_OS_IPHONE)
#include <crt_externs.h>
#endif
#endif

char*
rust_list_dir_val(struct dirent* entry_ptr) {
    return entry_ptr->d_name;
}

// Android's struct dirent does have d_type from the very beginning
// (android-3). _DIRENT_HAVE_D_TYPE is not defined all the way to android-21
// though...
#if defined(__ANDROID__)
# define _DIRENT_HAVE_D_TYPE
#endif

int
rust_dir_get_mode(struct dirent* entry_ptr) {
#if defined(_DIRENT_HAVE_D_TYPE) || defined(__APPLE__)
    switch (entry_ptr->d_type) {
        case DT_BLK: return S_IFBLK;
        case DT_CHR: return S_IFCHR;
        case DT_FIFO: return S_IFIFO;
        case DT_LNK: return S_IFLNK;
        case DT_REG: return S_IFREG;
        case DT_SOCK: return S_IFSOCK;
        case DT_DIR: return S_IFDIR;
    }
#endif
    return -1;
}

ino_t
rust_dir_get_ino(struct dirent* entry_ptr) {
    return entry_ptr->d_ino;
}

DIR*
rust_opendir(char *dirname) {
    return opendir(dirname);
}

int
rust_readdir_r(DIR *dirp, struct dirent *entry, struct dirent **result) {
    return readdir_r(dirp, entry, result);
}

size_t
rust_dirent_t_size() {
    return sizeof(struct dirent);
}

#if defined(__BSD__)
static int
get_num_cpus() {
    /* swiped from http://stackoverflow.com/questions/150355/
       programmatically-find-the-number-of-cores-on-a-machine */

    unsigned int numCPU;
    int mib[4];
    size_t len = sizeof(numCPU);

    /* set the mib for hw.ncpu */
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    /* get the number of CPUs from the system */
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if( numCPU < 1 ) {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if( numCPU < 1 ) {
            numCPU = 1;
        }
    }
    return numCPU;
}
#elif defined(__GNUC__)
static int
get_num_cpus() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}
#endif

uintptr_t
rust_get_num_cpus() {
    return get_num_cpus();
}

#if defined(__DragonFly__)
#include <errno.h>
// In DragonFly __error() is an inline function and as such
// no symbol exists for it.
int *__dfly_error(void) { return __error(); }
#endif

#if defined(__Bitrig__)
#include <stdio.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <limits.h>

int rust_get_path(void *p, size_t* sz)
{
  int mib[4];
  char *eq = NULL;
  char *key = NULL;
  char *val = NULL;
  char **menv = NULL;
  size_t maxlen, len;
  int nenv = 0;
  int i;

  if ((p == NULL) && (sz == NULL))
    return -1;

  /* get the argv array */
  mib[0] = CTL_KERN;
  mib[1] = KERN_PROC_ARGS;
  mib[2] = getpid();
  mib[3] = KERN_PROC_ENV;

  /* get the number of bytes needed to get the env */
  maxlen = 0;
  if (sysctl(mib, 4, NULL, &maxlen, NULL, 0) == -1)
    return -1;

  /* allocate the buffer */
  if ((menv = calloc(maxlen, sizeof(char))) == NULL)
    return -1;

  /* get the env array */
  if (sysctl(mib, 4, menv, &maxlen, NULL, 0) == -1)
  {
    free(menv);
    return -1;
  }

  mib[3] = KERN_PROC_NENV;
  len = sizeof(int);
  /* get the length of env array */
  if (sysctl(mib, 4, &nenv, &len, NULL, 0) == -1)
  {
    free(menv);
    return -1;
  }

  /* find _ key and resolve the value */
  for (i = 0; i < nenv; i++)
  {
    if ((eq = strstr(menv[i], "=")) == NULL)
      continue;

    key = menv[i];
    val = eq + 1;
    *eq = '\0';

    if (strncmp(key, "PATH", maxlen) != 0)
      continue;

    if (p == NULL)
    {
      /* return the length of the value + NUL */
      *sz = strnlen(val, maxlen) + 1;
      free(menv);
      return 0;
    }
    else
    {
      /* copy *sz bytes to the output buffer */
      memcpy(p, val, *sz);
      free(menv);
      return 0;
    }
  }

  free(menv);
  return -1;
}

int rust_get_path_array(void * p, size_t * sz)
{
  char *path, *str;
  char **buf;
  int i, num;
  size_t len;

  if ((p == NULL) && (sz == NULL))
    return -1;

  /* get the length of the PATH value */
  if (rust_get_path(NULL, &len) == -1)
    return -1;

  if (len == 0)
    return -1;

  /* allocate the buffer */
  if ((path = calloc(len, sizeof(char))) == NULL)
    return -1;

  /* get the PATH value */
  if (rust_get_path(path, &len) == -1)
  {
    free(path);
    return -1;
  }

  /* count the number of parts in the PATH */
  num = 1;
  for(str = path; *str != '\0'; str++)
  {
    if (*str == ':')
      num++;
  }

  /* calculate the size of the buffer for the 2D array */
  len = (num * sizeof(char*) + 1) + strlen(path) + 1;

  if (p == NULL)
  {
    free(path);
    *sz = len;
    return 0;
  }

  /* make sure we have enough buffer space */
  if (*sz < len)
  {
    free(path);
    return -1;
  }

  /* zero out the buffer */
  buf = (char**)p;
  memset(buf, 0, *sz);

  /* copy the data into the right place */
  str = p + ((num+1) * sizeof(char*));
  memcpy(str, path, strlen(path));

  /* parse the path into it's parts */
  for (i = 0; i < num && (buf[i] = strsep(&str, ":")) != NULL; i++) {;}
  buf[num] = NULL;

  free(path);
  return 0;
}

int rust_get_argv_zero(void* p, size_t* sz)
{
  int mib[4];
  char **argv = NULL;
  size_t len;

  if ((p == NULL) && (sz == NULL))
    return -1;

  /* get the argv array */
  mib[0] = CTL_KERN;
  mib[1] = KERN_PROC_ARGS;
  mib[2] = getpid();
  mib[3] = KERN_PROC_ARGV;

  /* request KERN_PROC_ARGV size */
  len = 0;
  if (sysctl(mib, 4, NULL, &len, NULL, 0) == -1)
    return -1;

  /* allocate buffer to receive the values */
  if ((argv = malloc(len)) == NULL)
    return -1;

  /* get the argv array */
  if (sysctl(mib, 4, argv, &len, NULL, 0) == -1)
  {
    free(argv);
    return -1;
  }

  /* get length of argv[0] */
  len = strnlen(argv[0], len) + 1;

  if (p == NULL)
  {
    *sz = len;
    free(argv);
    return 0;
  }

  if (*sz < len)
  {
    free(argv);
    return -1;
  }

  memcpy(p, argv[0], len);
  free(argv);
  return 0;
}

const char * rust_current_exe()
{
  static char *self = NULL;
  char *argv0;
  char **paths;
  size_t sz;
  int i;
  /* If `PATH_MAX` is defined on the platform, `realpath` will truncate the
   * resolved path up to `PATH_MAX`. While this can make the resolution fail if
   * the executable is placed in a deep path, the usage of a buffer whose
   * length depends on `PATH_MAX` is still memory safe. */
  char buf[2*PATH_MAX], exe[PATH_MAX];

  if (self != NULL)
    return self;

  if (rust_get_argv_zero(NULL, &sz) == -1)
    return NULL;
  if ((argv0 = calloc(sz, sizeof(char))) == NULL)
    return NULL;
  if (rust_get_argv_zero(argv0, &sz) == -1)
  {
    free(argv0);
    return NULL;
  }

  /* if argv0 is a relative or absolute path, resolve it with realpath */
  if ((*argv0 == '.') || (*argv0 == '/') || (strstr(argv0, "/") != NULL))
  {
    self = realpath(argv0, NULL);
    free(argv0);
    return self;
  }

  /* get the path array */
  if (rust_get_path_array(NULL, &sz) == -1)
  {
    free(argv0);
    return NULL;
  }
  if ((paths = calloc(sz, sizeof(char))) == NULL)
  {
    free(argv0);
    return NULL;
  }
  if (rust_get_path_array(paths, &sz) == -1)
  {
    free(argv0);
    free(paths);
    return NULL;
  }

  for(i = 0; paths[i] != NULL; i++)
  {
    snprintf(buf, 2*PATH_MAX, "%s/%s", paths[i], argv0);
    if (realpath(buf, exe) == NULL)
      continue;

    if (access(exe, F_OK | X_OK) == -1)
      continue;

    self = strdup(exe);
    free(argv0);
    free(paths);
    return self;
  }

  free(argv0);
  free(paths);
  return NULL;
}

#elif defined(__OpenBSD__)

#include <sys/param.h>
#include <sys/sysctl.h>
#include <limits.h>

const char * rust_current_exe() {
    static char *self = NULL;

    if (self == NULL) {
        int mib[4];
        char **argv = NULL;
        size_t argv_len;

        /* initialize mib */
        mib[0] = CTL_KERN;
        mib[1] = KERN_PROC_ARGS;
        mib[2] = getpid();
        mib[3] = KERN_PROC_ARGV;

        /* request KERN_PROC_ARGV size */
        argv_len = 0;
        if (sysctl(mib, 4, NULL, &argv_len, NULL, 0) == -1)
            return (NULL);

        /* allocate size */
        if ((argv = malloc(argv_len)) == NULL)
            return (NULL);

        /* request KERN_PROC_ARGV */
        if (sysctl(mib, 4, argv, &argv_len, NULL, 0) == -1) {
            free(argv);
            return (NULL);
        }

        /* get realpath if possible */
        if ((argv[0] != NULL) && ((*argv[0] == '.') || (*argv[0] == '/')
                                || (strstr(argv[0], "/") != NULL)))

            self = realpath(argv[0], NULL);
        else
            self = NULL;

        /* cleanup */
        free(argv);
    }

    return (self);
}

#endif

#endif // !defined(_WIN32)

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
