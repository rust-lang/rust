OS="`uname`"

cp sys/gsl_sys.h .
cp cblas/gsl_cblas.h .
rm cblas/test*.c
echo > config.h

case $OS in
  'Darwin') 
sed -i.bu "s/int/int64_t/g" cblas/*.c
sed -i.bu "s/gsl\//gsl64\//g" cblas/*.c cblas/*.h gsl_math.h gsl_precision.h gsl_pow_int.h gsl_minmax.h
sed -i.bu "s/cblas_\(.*\) (/cblas_\164_ (/g" cblas/*.c
sed -i.bu "1s/^/#include <stdint.h>\n/" cblas/*.c
    ;;
  *)
sed "s/int/int64_t/g" -i cblas/*.c
sed "s/gsl\//gsl64\//g" -i cblas/*.c cblas/*.h gsl_math.h gsl_precision.h gsl_pow_int.h gsl_minmax.h
sed "s/cblas_\(.*\) (/cblas_\164_ (/g" -i cblas/*.c
sed "1s/^/#include <stdint.h>\n/" -i cblas/*.c
;;
esac
