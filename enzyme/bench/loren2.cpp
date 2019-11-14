#include <iostream>
#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>

#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e){
    //do nothing
}

using namespace std;
using namespace boost::numeric::odeint;

const double sigma = 10.0;
const double R = 28.0;
const double b = 8.0 / 3.0;

#include <stdio.h>

typedef boost::array< double , 2 > state_type;

void lorenz( const state_type &x , state_type &dxdt , double t )
{
    const double gam = 0.15;
    dxdt[0] = x[1];
    dxdt[1] = -x[0] - gam*x[1];
    /*
    dxdt[0] = sigma * ( x[1] - x[0] );
    dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
    dxdt[2] = -b * x[2] + x[0] * x[1];
    */
}

void write_lorenz( const state_type &x , const double t )
{
    //cout << t << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << endl;
}
    
double foobar(double init=10.0) {
    state_type x = { init , 1.0 }; // initial conditions
    //state_type x = { init , 1.0 , 1.0 }; // initial conditions

    typedef controlled_runge_kutta< runge_kutta_dopri5< state_type , typename state_type::value_type , state_type , double > > stepper_type;
    integrate_const( stepper_type(), lorenz , x , 0.0 , 25.0 , 0.1 , write_lorenz );
    //integrate( lorenz , x , 0.0 , 25.0 , 0.1 , write_lorenz );
    printf("final result x[0]=%f x[1]=%f x[2]=%f\n", x[0], x[1], x[2]);
    return x[0];
}

extern "C" {
extern double __enzyme_autodiff(void*, double);
}

int main(int argc, char **argv)
{
    __enzyme_autodiff((void*)foobar, 3.4);
}
